from enum import Enum
import logging
import os
from pathlib import Path
import yaml

import typer
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.data.dataset import SeasonStratifiedIntensitySplit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

class SubDomainOption(str, Enum):
    london = "london"

@app.callback()
def callback():
    pass

@app.command()
def create(config: Path, input_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"), output_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"), val_prop: float = 0.2,
    test_prop: float = 0.1):
    """
    Create a dataset
    """
    config_name = config.basename
    config = yaml.load(config)

    predictand_var_params = {k: config[k] for k in ["domain", "ensemble_member", "scenario", "frequency"]}
    predictand_var_params.update({"variable": config["predictand"]["name"], "resolution":  config["predictand"]["resolution"]})
    predictand_meta = UKCPDatasetMetadata(input_base_dir, **predictand_var_params)

    predictors_meta = []
    for predictor_var_config in config["predictors"]:
        var_params = {k: config[k] for k in ["domain", "ensemble_member", "scenario", "frequency", "resolution"]}
        var_params.update({"variable": predictor_var_config["name"]})
        predictors_meta.append(UKCPDatasetMetadata(input_base_dir, **var_params))

    example_predictor_filepath = predictors_meta[0].existing_filepaths()[0]
    time_encoding = xr.open_dataset(example_predictor_filepath).time_bnds.encoding

    predictor_datasets = [xr.open_mfdataset(dsmeta.existing_filepaths()) for dsmeta in predictors_meta]
    predictand_dataset = xr.open_mfdataset(predictand_meta.existing_filepaths()).rename({predictand_meta.variable: f'target_{predictand_meta.variable}'})

    combined_dataset = xr.combine_by_coords([*predictor_datasets, predictand_dataset], compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all").isel(ensemble_member=0)
    combined_dataset = combined_dataset.assign_coords(season=(('time'), (combined_dataset.month_number.values % 12 // 3)))

    if config["split_scheme"]:
        train_set, val_set, test_set = SeasonStratifiedIntensitySplit(val_prop=val_prop, test_prop=test_prop, time_encoding=time_encoding).run(combined_dataset)
    else:
        raise(f"Unknown split scheme {config['split_scheme']}")

    output_subdir = "_".join([config["resolution"], config["domain"], config_name, config["split_scheme"]])
    output_dir = os.path.join(output_base_dir, "nc-datasets", output_subdir)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving data to {output_dir}")
    yaml.dump(config, os.path.join(output_dir, "ds-config.yml"))
    test_set.to_netcdf(os.path.join(output_dir, 'test.nc'))
    val_set.to_netcdf(os.path.join(output_dir, 'val.nc'))
    train_set.to_netcdf(os.path.join(output_dir, 'train.nc'))
