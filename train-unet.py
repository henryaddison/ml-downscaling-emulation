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
    parser = argparse.ArgumentParser(description='Regrid GCM data to match the CPM data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hi-res', dest='hires_file', type=Path, required=True,
                        help='Path to file containing 2.2km data')
    parser.add_argument('--lo-res', dest='lores_files', nargs='+', type=Path, required=True,
                        help='Paths to (interpolated) 60km data files')
    parser.add_argument('--model', dest='model_checkpoints_dir', type=Path, required=True,
                        help='Base path to storage for models')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')

    return parser.parse_args()

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

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    os.makedirs(args.model_checkpoints_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Prep data loaders
    unstacked_X = map(torch.tensor, map(np.load, args.lores_files))
    X = torch.stack(list(unstacked_X), dim=1)
    num_samples, num_predictors, _, _ = X.shape
    y = torch.tensor(np.load(args.hires_file)).unsqueeze(dim=1)

    all_data = TensorDataset(X, y)

    train_size = int(0.7 * len(all_data))
    val_size = len(all_data) - train_size
    train_set, val_set = random_split(all_data, [train_size, val_size])

    train_dl = DataLoader(train_set, batch_size=args.batch_size)
    val_dl = DataLoader(val_set, batch_size=args.batch_size)

    # Setup model, loss and optimiser
    model = unet.UNet(num_predictors, 1).to(device=device)

    criterion = torch.nn.L1Loss(reduction='mean').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Fit model
    for epoch in range(args.epochs):
        # Update model based on training data
        model.train()

        epoch_loss = 0.0

        for i, (batch_X, batch_y) in enumerate(train_dl):
            loss = train_on_batch(batch_X.to(device), batch_y.to(device), model)

            # Progress
            epoch_loss += loss.item()
            if (i+1) % (len(train_dl)//10) == 0:
                logging.info(f"Epoch {epoch}: Batch {i} Loss {loss.item()} Running epoch loss{epoch_loss}")

        # Compute validation loss
        model.eval()

        epoch_val_loss = 0
        for batch_X, batch_y in val_dl:
            val_loss = val_on_batch(batch_X.to(device), batch_y.to(device), model)

            # Progress
            epoch_val_loss += val_loss.item()

        model.train()

        logging.info(f"Epoch {epoch}: Loss {epoch_loss} Val loss {epoch_val_loss}")

        # Checkpoint model
        if (epoch % 10 == 0) or (epoch + 1 == args.epochs): # every 10th epoch or final one (to be safe)
            model_checkpoint_path = args.model_checkpoints_dir / f"model-epoch{epoch}.pth"
            torch.save(model, model_checkpoint_path)
            logging.info(f"Epoch {epoch}: Saved model to {model_checkpoint_path}")
