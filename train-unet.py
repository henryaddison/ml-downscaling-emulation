import argparse, logging, os, sys
from pathlib import Path
dir2 = os.path.abspath('unet/unet')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import unet
import torch

import numpy as np
import pandas as pd
import xarray as xr

# def dir_path(dir_arg):
#     return Path(dir_arg)

# type=dir_path

def get_args():
    parser = argparse.ArgumentParser(description='Regrid GCM data to match the CPM data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hi-res', dest='hires_dir', type=Path, required=True,
                        help='Base path to storage for 2.2km data')
    parser.add_argument('--lo-res', dest='lores_dir', type=Path, required=True,
                        help='Base path to storage for 60km data')
    parser.add_argument('--model', dest='model_checkpoints_dir', type=Path, required=True,
                        help='Base path to storage for models')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')

    return parser.parse_args()

def labels_from_xr(ds):
    return torch.from_numpy(ds.target_pr.values)

def inputs_tensor_from_xr(ds):
    return torch.stack((torch.from_numpy(ds.psl.values),  torch.from_numpy(ds.pr.values)), dim=1)

def train_on_batch(xr_batch, model, device):
    inputs_tensor = inputs_tensor_from_xr(xr_batch).to(device)
    labels_tensor = labels_from_xr(xr_batch).to(device)

    # Compute prediction and loss
    outputs_tensor = model(inputs_tensor)
    loss = criterion(outputs_tensor.squeeze(), labels_tensor)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def val_on_batch(xr_batch, model, device):
    with torch.no_grad():
        inputs_tensor = inputs_tensor_from_xr(xr_batch).to(device)
        labels_tensor = labels_from_xr(xr_batch).to(device)

        # Compute prediction and loss
        outputs_tensor = model(inputs_tensor)
        loss = criterion(outputs_tensor.squeeze(), labels_tensor)

    return loss

def predict_batch(xr_batch, model, device):
    inputs_tensor = inputs_tensor_from_xr(xr_batch).to(device)
    return  model(inputs_tensor)

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Prep data
    constraints = dict(ensemble_member=1)

    box_size = 32
    bl_long_idx = 250
    bl_lat_idx = 150

    latlong_ibox = {"grid_latitude": slice(bl_lat_idx, bl_lat_idx+box_size), "grid_longitude": slice(bl_long_idx, bl_long_idx+box_size)}

    hires_dir = Path(args.hires_dir)
    lores_dir = Path(args.lores_dir)

    cpmdata = xr.open_mfdataset(str(args.hires_dir / "*.nc")).rename({"pr": "target_pr"})
    cpmdata = cpmdata.loc[constraints]
    cpmdata = cpmdata.reset_coords()[['target_pr']]

    predictors = ["pr", "psl"]
    regridded_gcmdata = xr.open_mfdataset(str(args.lores_dir / "*/day/*.nc"))
    regridded_gcmdata = regridded_gcmdata.loc[constraints]
    regridded_gcmdata = regridded_gcmdata.reset_coords()[predictors]

    merged_data = xr.merge([regridded_gcmdata, cpmdata])

    # split training/test based on date
    training_data = merged_data.isel({"time": range(0,360)})
    validation_data = merged_data.isel({"time": range(360,540)})
    test_data = merged_data.isel({"time": range(540,720)})

    # Setup model, loss and optimiser
    model = unet.UNet(len(predictors), 1).to(device=device)

    criterion = torch.nn.L1Loss(reduction='mean').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Fit model
    for epoch in range(args.epochs):
        # Update model based on training data
        model.train()

        epoch_loss = 0.0
        for i in range(len(training_data.time)//args.batch_size):
            training_batch = training_data.isel(latlong_ibox).isel({"time": slice(i*args.batch_size, (i+1)*args.batch_size)})

            loss = train_on_batch(training_batch, model, device)

            # Progress
            epoch_loss += loss.item()
            if (i+1) % 30 == 0:
                logging.info(f"Epoch {epoch}: Batch {i} Loss {loss.item()} Running epoch loss{epoch_loss}")

        # Compute validation loss
        model.eval()

        epoch_val_loss = 0
        for i in range(len(validation_data.time)//args.batch_size):
            val_batch = validation_data.isel(latlong_ibox).isel({"time": slice(i*args.batch_size, (i+1)*args.batch_size)})
            val_loss = val_on_batch(val_batch, model, device)

            # Progress
            epoch_val_loss += val_loss.item()

        logging.info(f"Loss {epoch_loss} Val loss {epoch_val_loss}")
        model.train()

        # Checkpoint model
        model_checkpoint_path = args.model_checkpoints_dir / f"model-epoch{epoch}.pth"
        torch.save(model, model_checkpoint_path)
        logging.info(f"Epoch {epoch}: Saved model to {model_checkpoint_path}")
