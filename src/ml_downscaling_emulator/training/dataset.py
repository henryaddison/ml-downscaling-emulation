import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class CropT():
  def __init__(self, size):
    self.size = size

  def fit(self, train_ds):
    return self

  def transform(self, ds):
    return ds.isel(grid_longitude=slice(0, self.size),grid_latitude=slice(0, self.size))

class Standardize():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, train_ds):
    self.means = { var:  train_ds[var].mean().values for var in self.variables }
    self.stds = { var:  train_ds[var].std().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = (ds[var] - self.means[var])/self.stds[var]

    return ds

class MinMax():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, train_ds):
    self.maxs = { var:  train_ds[var].max().values for var in self.variables }
    self.mins = { var:  train_ds[var].min().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = (ds[var]-self.mins[var])/(self.maxs[var]-self.mins[var])

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]*(self.maxs[var]-self.mins[var]) + self.mins[var]

    return ds

class UnitRangeT():
  """WARNING: This transform assumes all values are positive"""
  def __init__(self, variables):
    self.variables = variables

  def fit(self, train_ds):
    self.maxs = { var:  train_ds[var].max().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]/self.maxs[var]

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]*self.maxs[var]

    return ds

class ClipT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, _train_ds):
    return self

  def transform(self, ds):
    # target pr should be all non-negative so transform is no-op
    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var].clip(min=0.0)

    return ds

class SqrtT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, _train_ds):
    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]**(0.5)

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]**2

    return ds

class ComposeT():
  def __init__(self, transforms):
    self.transforms = transforms

  def fit_transform(self, train_ds):
    for t in self.transforms:
      train_ds = t.fit(train_ds).transform(train_ds)

    return train_ds

  def transform(self, ds):
    for t in self.transforms:
      ds = t.transform(ds)

    return ds

  def invert(self, ds):
    for t in reversed(self.transforms):
      ds = t.invert(ds)

    return ds

class XRDataset(Dataset):
    def __init__(self, ds, variables):
        self.ds = ds
        self.variables = variables

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        subds = self.ds.isel(time=idx)

        cond = torch.tensor(np.stack([subds[var].values for var in self.variables], axis=0)).float()

        x = torch.tensor(np.stack([subds["target_pr"].values], axis=0)).float()

        return cond, x

def build_input_transform(variables, img_size):
  return ComposeT([
        CropT(img_size),
        Standardize(variables),
        UnitRangeT(variables)])

def build_target_transform(target_variables):
  return ComposeT([
        SqrtT(target_variables),
        ClipT(target_variables),
        UnitRangeT(target_variables),
      ])
