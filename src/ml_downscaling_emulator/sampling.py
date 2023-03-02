import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr

from mlde_utils.torch import XRDataset


def generate_np_samples(model, cond_batch):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cond_batch = cond_batch.to(device)

    samples = model(cond_batch)
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # extract numpy array
    samples = samples.cpu().detach().numpy()
    return samples


def np_samples_to_xr(samples, xr_eval_ds, target_transform):
    coords = {**dict(xr_eval_ds.coords)}

    cf_data_vars = {
        key: xr_eval_ds.data_vars[key]
        for key in [
            "rotated_latitude_longitude",
            "time_bnds",
            "grid_latitude_bnds",
            "grid_longitude_bnds",
        ]
    }

    pred_pr_dims = ["time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {
        "grid_mapping": "rotated_latitude_longitude",
        "standard_name": "pred_pr",
        "units": "kg m-2 s-1",
    }
    pred_pr_var = (pred_pr_dims, samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    pred_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={})

    if target_transform is not None:
        pred_ds = target_transform.invert(pred_ds)

    pred_ds = pred_ds.rename({"target_pr": "pred_pr"})

    return pred_ds


def sample(model, xr_data_eval, batch_size, variables, target_transform):
    np_samples = []
    with logging_redirect_tqdm():
        with tqdm(
            total=len(xr_data_eval["time"]), desc=f"Sampling", unit=" timesteps"
        ) as pbar:
            with torch.no_grad():
                for i in range(0, xr_data_eval["time"].shape[0], batch_size):
                    batch_times = xr_data_eval["time"][i : i + batch_size]
                    batch_ds = xr_data_eval.sel(time=batch_times)

                    cond_batch = XRDataset.to_tensor(batch_ds, variables)

                    batch_np_samples = generate_np_samples(model, cond_batch)
                    np_samples.append(batch_np_samples)

                    pbar.update(batch_np_samples.shape[0])

    # combine the samples along the time/batch axis
    np_samples = np.concatenate(np_samples, axis=0)

    xr_samples = np_samples_to_xr(np_samples, xr_data_eval, target_transform)

    return xr_samples
