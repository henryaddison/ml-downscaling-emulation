import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class XRDataset(Dataset):
    def __init__(self, ds, variables):
        self.ds = ds
        self.variables = variables

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        subds = self.ds.isel(time=idx, ensemble_member=0)

        X = torch.tensor(np.stack([subds[var].values for var in self.variables], axis=0))
        y = torch.tensor(np.stack([subds["target_pr"].values], axis=0))
        return X, y