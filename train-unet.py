import argparse
import logging
import os
from pathlib import Path
import sys

import torch
dir2 = os.path.abspath('unet/unet')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import unet

from training import train, load_data, log_epoch, track_run, checkpoint_model

ARCHITECTURE="U-Net"
EXPERIMENT_NAME="ml-downscaling-emulator"
TAGS = ["baseline", ARCHITECTURE]

def get_args():
    parser = argparse.ArgumentParser(description=f'Train {ARCHITECTURE} to downscale',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loss', '-l', dest='loss', type=str, default='l1', help='Loss function')
    parser.add_argument('--data', dest='data_dir', type=Path, required=True,
                        help='Path to directory of training and validation tensors')
    parser.add_argument('--model', dest='model_checkpoints_dir', type=Path, required=True,
                        help='Base path to storage for models')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=2e-4, help='Learning rate')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    logging.info(f"Starting {os.path.basename(__file__)}")

    os.makedirs(args.model_checkpoints_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Prep data loaders
    train_dl, val_dl = load_data(args.data_dir, args.batch_size)

    # Setup model, loss and optimiser
    num_predictors, _, _ = train_dl.dataset[0][0].shape
    model = unet.UNet(num_predictors, 1).to(device=device)

    if args.loss == 'l1':
        criterion = torch.nn.L1Loss().to(device)
    elif args.loss == 'mse':
        criterion = torch.nn.MSELoss().to(device)
    else:
        raise("Unkwown loss function")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_config = dict(
        dataset = args.data_dir,
        batch_size = train_dl.batch_size,
        epochs=args.epochs,
        architecture = args.arch,
        model_opts = {},
        loss = criterion.__class__.__name__,
        optimizer =  optimizer.__class__.__name__,
        optimizer_config = optimizer.defaults,
        device = device
    )

    with track_run(EXPERIMENT_NAME, run_config, TAGS) as (wandb_run, tb_writer):
        # Fit model
        wandb_run.watch(model, criterion=criterion, log_freq=100)
        for (epoch, epoch_metrics) in train(train_dl, val_dl, model, criterion, optimizer, args.epochs, device):
            log_epoch(epoch, epoch_metrics, wandb_run, tb_writer)

            # Checkpoint model
            # checkpoint the model about 10 times and the final one (to be safe)
            if (args.epochs <= 10) or (epoch % (args.epochs//10) == 0) or (epoch + 1 == args.epochs):
                checkpoint_model(model, args.model_checkpoints_dir, epoch)

    logging.info(f"Finished {os.path.basename(__file__)}")
