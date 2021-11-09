import argparse, logging, os, sys
from pathlib import Path
dir2 = os.path.abspath('unet/unet')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import unet
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

import numpy as np

import wandb
from mlflow import log_metric, log_param, log_artifacts, set_experiment, set_tags
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Train U-Net to downscale',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loss', '-l', dest='loss', type=str, default='l1', help='Loss function')
    parser.add_argument('--data', dest='data_dir', type=Path, required=True,
                        help='Path to directory of training and validation tensors')
    parser.add_argument('--model', dest='model_checkpoints_dir', type=Path, required=True,
                        help='Base path to storage for models')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')

    return parser.parse_args()

def train_epoch(model, dataloader, device, epoch):
    model.train()

    epoch_loss = 0.0

    for i, (batch_X, batch_y) in enumerate(dataloader):
        loss = train_on_batch(batch_X.to(device), batch_y.to(device), model)
        epoch_loss += loss.item()

        # Log progress at least every 10th batch
        if (len(dataloader) <= 10) or ((i+1) % max(len(dataloader)//10,1) == 0):
            logging.info(f"Epoch {epoch}: Batch {i}: Batch Train Loss {loss.item()} Running Epoch Train Loss {epoch_loss}")

    return epoch_loss

def val_epoch(model, dataloader, device, epoch):
    model.eval()

    epoch_val_loss = 0
    for batch_X, batch_y in dataloader:
        val_loss = val_on_batch(batch_X.to(device), batch_y.to(device), model)

        # Progress
        epoch_val_loss += val_loss.item()

    model.train()

    return epoch_val_loss

def train_on_batch(batch_X, batch_y, model):
    # Compute prediction and loss
    outputs_tensor = model(batch_X)
    loss = criterion(outputs_tensor, batch_y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def val_on_batch(batch_X, batch_y, model):
    with torch.no_grad():
        # Compute prediction and loss
        outputs_tensor = model(batch_X)
        loss = criterion(outputs_tensor, batch_y)

    return loss

def load_data(data_dirpath, batch_size):
    train_set = TensorDataset(torch.load(data_dirpath/'train_X.pt'), torch.load(data_dirpath/'train_y.pt'))
    val_set = TensorDataset(torch.load(data_dirpath/'val_X.pt'), torch.load(data_dirpath/'val_y.pt'))

    train_dl = DataLoader(train_set, batch_size=batch_size)
    val_dl = DataLoader(val_set, batch_size=batch_size)

    return train_dl, val_dl

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

    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    config = dict(
        dataset = args.data_dir,
        optimizer = "Adam",
        learning_rate = learning_rate,
        batch_size = args.batch_size,
        architecture = "U-Net",
        device = device,
        epochs=args.epochs
    )

    wandb.init(
        project="ml-downscaling-emulator",
        tags=["baseline", "U-Net"],
        config=config
    )

    wandb.watch(model, criterion=criterion, log_freq=100)

    set_experiment("ml-downscaling-emulator")
    set_tags({"model": "U-Net", "purpose": "baseline"})
    log_param("dataset", args.data_dir)
    log_param("optimizer", "Adam")
    log_param("learning_rate", learning_rate)
    log_param("batch_size", args.batch_size)
    log_param("architecture", "U-Net")
    log_param("device", device)
    log_param("epochs", args.epochs)

    writer = SummaryWriter()

    # Fit model
    for epoch in range(args.epochs):
        # Update model based on training data
        epoch_train_loss = train_epoch(model, train_dl, device, epoch)

        # Compute validation loss
        epoch_val_loss = val_epoch(model, val_dl, device, epoch)

        logging.info(f"Epoch {epoch}: Train Loss {epoch_train_loss} Val Loss {epoch_val_loss}")
        wandb.log({"train/loss": epoch_train_loss, "val/loss": epoch_val_loss})
        log_metric("train/loss",epoch_train_loss, step=epoch)
        log_metric("val/loss", epoch_val_loss, step=epoch)
        writer.add_scalar("train/loss", epoch_train_loss, epoch)
        writer.add_scalar("val/loss", epoch_val_loss, epoch)
        # Checkpoint model
        if (epoch % 10 == 0) or (epoch + 1 == args.epochs): # every 10th epoch or final one (to be safe)
            model_checkpoint_path = args.model_checkpoints_dir / f"model-epoch{epoch}.pth"
            torch.save(model, model_checkpoint_path)
            logging.info(f"Epoch {epoch}: Saved model to {model_checkpoint_path}")

    # writer.add_hparams(config, {"train/loss": epoch_train_loss, "val/loss": epoch_val_loss})
    writer.flush()
    logging.info(f"Finished {os.path.basename(__file__)}")
