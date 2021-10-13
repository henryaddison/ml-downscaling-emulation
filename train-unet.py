import argparse, logging, os, sys
dir2 = os.path.abspath('unet/unet')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import unet
import torch

import numpy as np
import pandas as pd
import xarray as xr

def get_args():
    parser = argparse.ArgumentParser(description='Regrid GCM data to match the CPM data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hi-res', dest='hires_dir', type=str, required=True,
                        help='Base path to storage for 2.2km data')
    parser.add_argument('--lo-res', dest='lores_dir', type=str, required=True,
                        help='Base path to storage for 60km data')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')

    return parser.parse_args()

def labels_from_xr(ds):
    return torch.from_numpy(ds.target_pr.values)

def inputs_tensor_from_xr(ds):
    return torch.stack((torch.from_numpy(ds.psl.values),  torch.from_numpy(ds.pr.values)), dim=1)

def train_on_batch(xr_batch, model):
    inputs_tensor = inputs_tensor_from_xr(xr_batch)
    labels_tensor = labels_from_xr(xr_batch)

    # Compute prediction and loss
    outputs_tensor = model(inputs_tensor)
    loss = criterion(outputs_tensor.squeeze(), labels_tensor)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def val_on_batch(xr_batch, model):
    with torch.no_grad():
        inputs_tensor = inputs_tensor_from_xr(xr_batch)
        labels_tensor = labels_from_xr(xr_batch)

        # Compute prediction and loss
        outputs_tensor = model(inputs_tensor)
        loss = criterion(outputs_tensor.squeeze(), labels_tensor)

    return loss

def predict_batch(xr_batch, model):
    inputs_tensor = inputs_tensor_from_xr(xr_batch)
    return  model(inputs_tensor)



if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')

    model = unet.UNet(2, 1)

    criterion = torch.nn.L1Loss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    predictors = ["pr", "psl"]
    constraints = dict(ensemble_member=1, time=slice("1980-12-01","1982-11-30"))

    box_size = 32
    bl_long_idx = 250
    bl_lat_idx = 150

    latlong_ibox = {"grid_latitude": slice(bl_lat_idx, bl_lat_idx+box_size), "grid_longitude": slice(bl_long_idx, bl_long_idx+box_size)}

    cpmdata = xr.open_mfdataset(f"{args.hires_dir}/pr/day/*.nc").rename({"pr": "target_pr"})
    cpmdata = cpmdata.loc[constraints]
    cpmdata = cpmdata.reset_coords()[['target_pr']]

    regridded_gcmdata = xr.open_mfdataset(f"{args.lores_dir}/*/day/*.nc")
    regridded_gcmdata = regridded_gcmdata.loc[constraints]
    regridded_gcmdata = regridded_gcmdata.reset_coords()[predictors]

    merged_data = xr.merge([regridded_gcmdata, cpmdata])
    # select a small subset of the data for trial purposes
    # merged_data = merged_data.isel({"grid_latitude": slice(512), "grid_longitude": slice(512)})

    # split training/test based on date
    training_data = merged_data.sel({"time": slice("1980-12-01", "1981-11-30")})
    validation_data = merged_data.sel({"time": slice("1981-12-01", "1982-05-30")})
    test_data = merged_data.sel({"time": slice("1982-06-01", "1982-11-30")})


    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        for i in range(len(training_data.time)//args.batch_size):
            training_batch = training_data.isel(latlong_ibox).isel({"time": slice(i*batch_size, (i+1)*batch_size)})

            loss = train_on_batch(training_batch, model)

            # Progress
            epoch_loss += loss.item()
            if (i+1) % 30 == 0:
                print(f"Epoch {epoch}: Batch {i} Loss {loss.item()}")

        model.eval()

        epoch_val_loss = 0
        for i in range(len(validation_data.time)//batch_size):
            val_batch = validation_data.isel(latlong_ibox).isel({"time": slice(i*batch_size, (i+1)*batch_size)})
            val_loss = val_on_batch(val_batch, model)

            # Progress
            epoch_val_loss += val_loss.item()

        print(f"Epoch {epoch}: Loss {epoch_loss} Val loss {epoch_val_loss}")

        model.train()


    torch.save(model, 'model.pth')
