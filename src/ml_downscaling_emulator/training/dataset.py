import os
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class CropT():
  def __init__(self, size):
    self.size = size

  def fit(self, target_ds, model_src_ds):
    return self

  def transform(self, ds):
    return ds.isel(grid_longitude=slice(0, self.size),grid_latitude=slice(0, self.size))

class Standardize():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    self.means = { var:  target_ds[var].mean().values for var in self.variables }
    self.stds = { var:  target_ds[var].std().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = (ds[var] - self.means[var])/self.stds[var]

    return ds

class GlobalStandardize():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    self.means = { var:  model_src_ds[var].mean().values for var in self.variables }
    self.stds = { var:  model_src_ds[var].std().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = (ds[var] - self.means[var])/self.stds[var]

    return ds


class SpatialStandardize():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    self.means = { var:  target_ds[var].mean(dim=["time"]).values for var in self.variables }
    self.stds = { var:  target_ds[var].std(dim=["time"]).values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = (ds[var] - self.means[var])/self.stds[var]

    return ds

class NoopT():
  def fit(self, target_ds, model_src_ds):
    return self

  def transform(self, ds):
    return ds

  def invert(self, ds):
    return ds

class MassageStats():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    self.target_means = { variable: target_ds[variable].mean(dim=["time"]) for variable in self.variables }
    self.target_stds = { variable: target_ds[variable].std(dim=["time"]) for variable in self.variables }

    self.model_src_means = { variable: model_src_ds[variable].mean(dim=["time"]) for variable in self.variables }
    self.model_src_stds = { variable: model_src_ds[variable].std(dim=["time"]) for variable in self.variables }

    return self

  def transform(self, ds):
    for variable in self.variables:
      ds[variable] = (ds[variable] - self.target_means[variable])*(self.model_src_stds[variable]/self.target_stds[variable]) + self.model_src_means[variable]
    return ds

class MinMax():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    self.maxs = { var:  target_ds[var].max().values for var in self.variables }
    self.mins = { var:  target_ds[var].min().values for var in self.variables }

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

  def fit(self, target_ds, model_src_ds):
    self.maxs = { var:  target_ds[var].max().values for var in self.variables }

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

  def fit(self, target_ds, model_src_ds):
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

  def fit(self, target_ds, model_src_ds):
    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]**(0.5)

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]**2

    return ds

class RootT():
  def __init__(self, variables, root_base):
    self.variables = variables
    self.root_base = root_base

  def fit(self, target_ds, model_src_ds):
    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = np.power(ds[var], 1/self.root_base)

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = np.power(ds[var], self.root_base)

    return ds

class LogT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = np.log1p(ds[var])

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = np.expm1(ds[var])

    return ds

class ComposeT():
  def __init__(self, transforms):
    self.transforms = transforms

  def fit(self, target_ds, model_src_ds=None):
    for t in self.transforms:
      target_ds = t.fit(target_ds, model_src_ds).transform(target_ds)

    return self

  def transform(self, ds):
    for t in self.transforms:
      ds = t.transform(ds)

    return ds

  def invert(self, ds):
    for t in reversed(self.transforms):
      ds = t.invert(ds)

    return ds

class XRDataset(Dataset):
    def __init__(self, ds, variables, target_variables):
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        subds = self.ds.isel(time=idx)

        cond = torch.tensor(np.stack([subds[var].values for var in self.variables], axis=0)).float()

        x = torch.tensor(np.stack([subds[var].values for var in self.target_variables], axis=0)).float()

        return cond, x

def build_input_transform(variables, key="v1"):
  if key == "v1":
    return ComposeT([
      Standardize(variables),
      UnitRangeT(variables)
    ])

  if key == "standardize":
    return ComposeT([
      Standardize(variables)
    ])

  if key == "massage":
    return ComposeT([
      MassageStats(variables),
      GlobalStandardize(variables),
    ])

  if key == "spatial":
    return ComposeT([
      SpatialStandardize(variables)
    ])

  raise(f"Unknown input transform {key}")

def build_target_transform(target_variables, key="v1"):
  if key == "v1":
    return ComposeT([
      SqrtT(target_variables),
      ClipT(target_variables),
      UnitRangeT(target_variables),
    ])

  if key == "sqrt":
    return ComposeT([
      RootT(target_variables, 2),
      ClipT(target_variables),
    ])

  if key == "cbrt":
    return ComposeT([
      RootT(target_variables, 3),
      ClipT(target_variables),
    ])

  if key == "qdrt":
    return ComposeT([
      RootT(target_variables, 4),
      ClipT(target_variables),
    ])

  if key == "log":
    return ComposeT([
      LogT(target_variables),
      ClipT(target_variables),
    ])

  raise(f"Unknown input transform {key}")

def get_variables(dataset_name):
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', dataset_name)
  with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
      ds_config = yaml.safe_load(f)

  variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]
  target_variables = ["target_pr"]

  return variables, target_variables
