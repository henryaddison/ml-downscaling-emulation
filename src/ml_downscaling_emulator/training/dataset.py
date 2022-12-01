from datetime import timedelta
import logging
import os
import pickle
import yaml

from flufl.lock import Lock
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

class PixelStandardize():
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

class PixelMatchModelSrcStandardize():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    self.pixel_target_means = { variable: target_ds[variable].mean(dim=["time"]) for variable in self.variables }
    self.pixel_target_stds = { variable: target_ds[variable].std(dim=["time"]) for variable in self.variables }

    self.pixel_model_src_means = { variable: model_src_ds[variable].mean(dim=["time"]) for variable in self.variables }
    self.pixel_model_src_stds = { variable: model_src_ds[variable].std(dim=["time"]) for variable in self.variables }

    self.global_model_src_means = { var:  model_src_ds[var].mean().values for var in self.variables }
    self.global_model_src_stds = { var:  model_src_ds[var].std().values for var in self.variables }

    return self

  def transform(self, ds):
    for variable in self.variables:
      # first standardize each pixel
      da_pixel_stan = (ds[variable] - self.pixel_target_means[variable])/self.pixel_target_stds[variable]
      # then match mean and variance of each pixel to model source distribution
      da_pixel_like_model_src = (da_pixel_stan * self.pixel_model_src_stds[variable]) + self.pixel_model_src_means[variable]
      # finally standardize globally (assuming a model source distribution)
      da_global_stan_like_model_src = (da_pixel_like_model_src - self.global_model_src_means[variable]) / self.global_model_src_stds[variable]
      ds[variable] = da_global_stan_like_model_src
    return ds

  def transform(self, ds):
    for variable in self.variables:
      # first standardize each pixel
      da_pixel_stan = (ds[variable] - self.pixel_target_means[variable])/self.pixel_target_stds[variable]
      # then match mean and variance of each pixel to model source distribution
      da_pixel_like_model_src = (da_pixel_stan * self.pixel_model_src_stds[variable]) + self.pixel_model_src_means[variable]
      # finally standardize globally (assuming a model source distribution)
      da_global_stan_like_model_src = (da_pixel_like_model_src - self.global_model_src_means[variable]) / self.global_model_src_stds[variable]
      ds[variable] = da_global_stan_like_model_src
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

class RecentreT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, target_ds, model_src_ds):
    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]*2. - 1.

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = (ds[var] + 1.) / 2.

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

class RawMomentT():
  def __init__(self, variables, root_base):
    self.variables = variables
    self.root_base = root_base

  def fit(self, target_ds, model_src_ds):
    self.raw_moments = { var: np.power(np.mean(np.power(target_ds[var], self.root_base)), 1/self.root_base) for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]/self.raw_moments[var]

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]*self.raw_moments[var]

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

  if key in ["standardize", "stan"]:
    return ComposeT([
      Standardize(variables)
    ])

  if key == "stanur":
    return ComposeT([
      Standardize(variables),
      UnitRangeT(variables),
    ])

  if key == "stanurrecen":
    return ComposeT([
      Standardize(variables),
      UnitRangeT(variables),
      RecentreT(variables),
    ])

  if key == "pixelstan":
    return ComposeT([
      PixelStandardize(variables)
    ])

  if key == "pixelmmsstan":
    return ComposeT([
      PixelMatchModelSrcStandardize(variables),
    ])

  if key == "pixelmmsstanur":
    return ComposeT([
      PixelMatchModelSrcStandardize(variables),
      UnitRangeT(variables),
    ])

  raise RuntimeError(f"Unknown input transform {key}")

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

  if key == "sqrtur":
    return ComposeT([
      RootT(target_variables, 2),
      ClipT(target_variables),
      UnitRangeT(target_variables),
    ])

  if key == "sqrturrecen":
    return ComposeT([
      RootT(target_variables, 2),
      ClipT(target_variables),
      UnitRangeT(target_variables),
      RecentreT(target_variables),
    ])

  if key == "sqrtrm":
    return ComposeT([
      RootT(target_variables, 2),
      RawMomentT(target_variables, 2),
      ClipT(target_variables),
    ])

  if key == "cbrt":
    return ComposeT([
      RootT(target_variables, 3),
      ClipT(target_variables),
    ])

  if key == "cbrtur":
    return ComposeT([
      RootT(target_variables, 3),
      ClipT(target_variables),
      UnitRangeT(target_variables),
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

  raise RuntimeError(f"Unknown input transform {key}")

def get_variables(dataset_name):
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', dataset_name)
  with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
      ds_config = yaml.safe_load(f)

  variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]
  target_variables = ["target_pr"]

  return variables, target_variables

def create_transform(variables, active_dataset_name, model_src_dataset_name, transform_key, builder, store_path):
  logging.info(f"Fitting transform")
  model_src_training_split = load_raw_dataset_split(model_src_dataset_name,"train")

  active_dataset_training_split = load_raw_dataset_split(active_dataset_name,"train")

  xfm = builder(variables, key=transform_key)

  xfm.fit(active_dataset_training_split, model_src_training_split)

  save_transform(xfm, store_path)

  return xfm

def save_transform(xfm, path):
  with open(path, 'wb') as f:
        logging.info(f"Storing transform: {path}")
        pickle.dump(xfm, f, pickle.HIGHEST_PROTOCOL)

def load_transform(path):
  with open(path, 'rb') as f:
    logging.info(f"Using stored transform: {path}")
    xfm = pickle.load(f)

  return xfm

def find_or_create_transforms(active_dataset_name, model_src_dataset_name, transform_dir, input_transform_key, target_transform_key, evaluation):
  dataset_transform_dir = os.path.join(transform_dir, active_dataset_name)
  os.makedirs(dataset_transform_dir, exist_ok=True)
  input_transform_path = os.path.join(dataset_transform_dir, 'input.pickle')
  target_transform_path = os.path.join(transform_dir, 'target.pickle')

  variables, target_variables = get_variables(model_src_dataset_name)

  lock_path = os.path.join(transform_dir, '.lock')
  lock = Lock(lock_path, lifetime=timedelta(hours=1))
  with lock:
    if os.path.exists(input_transform_path):
      input_transform = load_transform(input_transform_path)
    else:

      input_transform = create_transform(variables, active_dataset_name, model_src_dataset_name, input_transform_key, build_input_transform, input_transform_path)

    if os.path.exists(target_transform_path):
      target_transform = load_transform(target_transform_path)
    else:
      if evaluation:
        raise RuntimeError("Target transform should only be fitted during training")
      target_transform = create_transform(target_variables, active_dataset_name, model_src_dataset_name, target_transform_key, build_target_transform, target_transform_path)

  return input_transform, target_transform


def load_raw_dataset_split(dataset_name, split):
    data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets',dataset_name)
    return xr.load_dataset(os.path.join(data_dirpath, f'{split}.nc'))

def get_dataset(active_dataset_name, model_src_dataset_name, input_transform_key, target_transform_key, transform_dir, batch_size, split, evaluation=False):
  """Create data loaders for given split.

  Args:
    active_dataset_name: Name of dataset from which to load data splits
    model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
    transform_dir: Path to where transforms should be stored
    input_transform_key: Name of input transform pipeline to use
    target_transform_key: Name of target transform pipeline to use
    batch_size: Size of batch to use for DataLoaders
    evaluation: If `True`, fix number of epochs to 1.
    split: Split of the active dataset to load

  Returns:
    data_loader, transform, target_transform.
  """

  transform, target_transform = find_or_create_transforms(active_dataset_name, model_src_dataset_name, transform_dir, input_transform_key, target_transform_key, evaluation)

  xr_data = load_raw_dataset_split(active_dataset_name, split)

  variables, target_variables = get_variables(model_src_dataset_name)

  xr_data = transform.transform(xr_data)
  xr_data = target_transform.transform(xr_data)
  xr_dataset = XRDataset(xr_data, variables, target_variables)
  data_loader = DataLoader(xr_dataset, batch_size=batch_size)

  return data_loader, transform, target_transform
