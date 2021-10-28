import argparse, logging, os, sys
from pathlib import Path
dir2 = os.path.abspath('unet/unet')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import unet
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Train U-Net to downscale',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loss', '-l', dest='loss', type=str, default='l1', help='Loss function')
    parser.add_argument('--hi-res', dest='hires_file', type=Path, required=True,
                        help='Path to file containing 2.2km data')
    parser.add_argument('--lo-res', dest='lores_files', nargs='+', type=Path, required=True,
                        help='Paths to (interpolated) 60km data files')
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

def load_data(lores_files, hires_file):
    unstacked_X = map(torch.tensor, map(np.load, lores_files))
    X = torch.stack(list(unstacked_X), dim=1)

    y = torch.tensor(np.load(hires_file)).unsqueeze(dim=1)

    all_data = TensorDataset(X, y)

    train_size = int(0.7 * len(all_data))
    val_size = len(all_data) - train_size
    train_set, val_set = random_split(all_data, [train_size, val_size])

    train_dl = DataLoader(train_set, batch_size=args.batch_size)
    val_dl = DataLoader(val_set, batch_size=args.batch_size)

    return train_dl, val_dl

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    logging.info(f"Starting {os.path.basename(__file__)}")

    os.makedirs(args.model_checkpoints_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Prep data loaders
    train_dl, val_dl = load_data(args.lores_files, args.hires_file)

    # Setup model, loss and optimiser
    num_predictors, _, _ = train_dl.dataset.dataset[0][0].shape
    model = unet.UNet(num_predictors, 1).to(device=device)

    if args.loss == 'l1':
        criterion = torch.nn.L1Loss(reduction='mean').to(device)
    elif args.loss == 'mse':
        criterion = torch.nn.MSELoss().to(device)
    else:
        raise("Unkwown loss function")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Fit model
    for epoch in range(args.epochs):
        # Update model based on training data
        epoch_train_loss = train_epoch(model, train_dl, device, epoch)

        # Compute validation loss
        epoch_val_loss = val_epoch(model, val_dl, device, epoch)

        logging.info(f"Epoch {epoch}: Train Loss {epoch_train_loss} Val Loss {epoch_val_loss}")

        # Checkpoint model
        if (epoch % 10 == 0) or (epoch + 1 == args.epochs): # every 10th epoch or final one (to be safe)
            model_checkpoint_path = args.model_checkpoints_dir / f"model-epoch{epoch}.pth"
            torch.save(model, model_checkpoint_path)
            logging.info(f"Epoch {epoch}: Saved model to {model_checkpoint_path}")

    logging.info(f"Finished {os.path.basename(__file__)}")
