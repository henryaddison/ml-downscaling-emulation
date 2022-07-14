import glob
from importlib_resources import files
import logging
import os
from pathlib import Path
import shutil
import subprocess
import yaml

import iris
import typer
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.bin import DomainOption, CollectionOption
from ml_downscaling_emulator.data.moose import VARIABLE_CODES, select_query, moose_path
from ml_downscaling_emulator.preprocessing.coarsen import Coarsen
from ml_downscaling_emulator.preprocessing.constrain import Constrain
from ml_downscaling_emulator.preprocessing.regrid import Regrid
from ml_downscaling_emulator.preprocessing.resample import Resample
from ml_downscaling_emulator.preprocessing.select_domain import SelectDomain
from ml_downscaling_emulator.preprocessing.sum import Sum
from ml_downscaling_emulator.preprocessing.vorticity import Vorticity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

@app.callback()
def callback():
    pass

def moose_extract_dirpath(variable: str, year: int, frequency: str, resolution: str, collection: str, domain: str):
    return Path(os.getenv("MOOSE_DATA"))/"pp"/collection/domain/resolution/"rcp85"/"01"/variable/frequency/str(year)

def ppdata_dirpath(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str):
    return moose_extract_dirpath(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution, collection=collection)/"data"

def nc_filename(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str):
    return f"{variable}_rcp85_{collection}_{domain}_{resolution}_01_{frequency}_{year-1}1201-{year}1130.nc"

def raw_nc_filepath(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str = "land-cpm"):
    return Path(os.getenv("MOOSE_DATA"))/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution, collection=collection)

def processed_nc_filepath(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str):
    return Path(os.getenv("DERIVED_DATA"))/"moose"/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution, collection=collection)

@app.command()
def extract(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", collection: CollectionOption = typer.Option(...)):
    """
    Extract data from moose
    """
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"

    query = select_query(year=year, variable=variable, frequency=frequency, collection=collection.value)

    output_dirpath = moose_extract_dirpath(variable=variable, year=year, frequency=frequency, resolution=resolution, collection=collection.value, domain=domain)
    query_filepath = output_dirpath/"searchfile"
    pp_dirpath = ppdata_dirpath(variable=variable, year=year, frequency=frequency, resolution=resolution, collection=collection.value, domain=domain)

    os.makedirs(output_dirpath, exist_ok=True)
    # remove any previous attempt at extracting the data (or else moo select will complain)
    shutil.rmtree(pp_dirpath, ignore_errors=True)
    os.makedirs(pp_dirpath, exist_ok=True)

    logger.debug(query)
    query_filepath.write_text(query)

    moose_uri = moose_path(variable, year, frequency=frequency, collection=collection.value)

    query_cmd = ["moo" , "select", query_filepath, moose_uri, os.path.join(pp_dirpath,"")]

    logger.debug(f"Running {query_cmd}")
    logger.info(f"Extracting {variable} for {year}...")

    output = subprocess.run(query_cmd, capture_output=True, check=True)
    stdout = output.stdout.decode("utf8")
    print(stdout)
    print(output.stderr.decode("utf8"))
    # os.execvp(query_cmd[0], query_cmd)

@app.command()
def convert(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", collection: CollectionOption = typer.Option(...)):
    """
    Convert pp data to a netCDF file
    """
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"

    pp_files_glob = ppdata_dirpath(variable=variable, year=year, frequency=frequency, resolution=resolution, collection=collection.value, domain=domain)/"*.pp"
    output_filepath = raw_nc_filepath(variable=variable, year=year, frequency=frequency, resolution=resolution, collection=collection.value, domain=domain)

    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)

    src_cube = iris.load(str(pp_files_glob))

    # bug in the xwind and ywind data means the final grid_latitude bound is very large (1.0737418e+09)
    if collection == CollectionOption.cpm and variable in ["xwind", "ywind"]:
        src_cube.coord("grid_latitude").bounds[-1][1] = 8.962849

    iris.save(src_cube, output_filepath)

    assert len(xr.open_dataset(output_filepath).time) == 360

@app.command()
def preprocess(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", scale_factor: int = typer.Option(...), subdomain: DomainOption = DomainOption.london, target_frequency: str = "day"):
    """
    Pre-process the moose data:
        1. Re-name the variable
        2. Re-sample the data to a match a target frequency
        3. Coarsen by given scale-factor
        4. Select a subdomain
    """
    input_filepath = raw_nc_filepath(variable=variable, year=year, frequency=frequency)

    ds = xr.load_dataset(input_filepath)

    if "moose_name" in VARIABLE_CODES[variable]:
        logger.info(f"Renaming {VARIABLE_CODES[variable]['moose_name']} to {variable}...")
        ds = ds.rename({VARIABLE_CODES[variable]["moose_name"]: variable})

    if frequency != target_frequency:
        ds = Resample(target_frequency=target_frequency).run(ds)

    if scale_factor != 1:
        typer.echo(f"Coarsening {scale_factor}x...")
        target_resolution = f"2.2km-coarsened-{scale_factor}x"
        uncoarsened_ds = ds.clone()
        ds = Coarsen(scale_factor=scale_factor).run(ds)
        ds = Regrid(target_grid=uncoarsened_ds, variable=variable).run(ds)
    else:
        target_resolution = "2.2km"

    typer.echo(f"Select {subdomain.value} subdomain...")
    ds = SelectDomain(subdomain=subdomain.value).run(ds)

    assert len(ds.grid_latitude) == 64
    assert len(ds.grid_longitude) == 64

    output_filepath = processed_nc_filepath(variable=variable, year=year, frequency=target_frequency, domain=subdomain.value, resolution=target_resolution)
    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    ds.to_netcdf(output_filepath)

@app.command()
def clean(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", collection: CollectionOption = typer.Option(...)):
    """
    Remove any unneccessary files once processing is done
    """
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"

    pp_path = ppdata_dirpath(variable=variable, year=year, frequency=frequency, collection=collection.value, resolution=resolution, domain=domain)
    typer.echo(f"Removing {pp_path}...")
    shutil.rmtree(pp_path, ignore_errors=True)
    raw_nc_path = raw_nc_filepath(variable=variable, year=year, frequency=frequency, collection=collection.value, resolution=resolution, domain=domain)
    typer.echo(f"Removing {raw_nc_path}...")
    os.remove(raw_nc_path)

@app.command()
def create_variable(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", domain: DomainOption = DomainOption.london, scenario="rcp85", scale_factor: str = typer.Option(...), target_resolution: str = "2.2km"):
    """
    Create a new variable from moose data
    """
    config = files('ml_downscaling_emulator.config').joinpath(f'variables/day/{variable}.yml').read_text()
    config = yaml.safe_load(config)

    variable = config['variable']
    collection = CollectionOption(config['sources']['collection'])
    if collection == CollectionOption.cpm:
        variable_resolution = "2.2km"
        source_domain = "uk"
    elif collection == CollectionOption.gcm:
        variable_resolution = "60km"
        source_domain = "global"
    else:
        raise f"Unknown collection {collection}"

    sources = {}

    # ds = xr.open_mfdataset([raw_nc_filepath(variable=source, year=year, frequency=frequency) for source in config['sources']['moose']])
    # for source in config['sources']['moose']:
    #     if "moose_name" in VARIABLE_CODES[source]:
    #         logger.info(f"Renaming {VARIABLE_CODES[source]['moose_name']} to {source}...")
    #         ds = ds.rename({VARIABLE_CODES[source]["moose_name"]: source})

    # currently only support data from moose
    assert(config["sources"]["type"] == "moose")

    for src_variable in config['sources']['variables']:
        source_nc_filepath = raw_nc_filepath(variable=src_variable, year=year, frequency=frequency, resolution=variable_resolution, collection=collection.value, domain=source_domain)
        logger.info(f"Opening {source_nc_filepath}")
        ds = xr.open_dataset(source_nc_filepath)

        if "moose_name" in VARIABLE_CODES[src_variable]:
            logger.info(f"Renaming {VARIABLE_CODES[src_variable]['moose_name']} to {src_variable}...")
            ds = ds.rename({VARIABLE_CODES[src_variable]["moose_name"]: src_variable})

        sources[src_variable] = ds

    logger.info(f"Combining {config['sources']}...")
    ds = xr.combine_by_coords(sources.values(), compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all")

    for job_spec in config['spec']:
        if job_spec['action'] == "sum":
            logger.info(f"Summing {job_spec['variables']}")
            ds = Sum(job_spec['variables'], config['variable']).run(ds)
            ds[config['variable']] = ds[config['variable']].assign_attrs(config['attrs'])
        elif job_spec['action'] == "coarsen":
            if scale_factor == "gcm":
                typer.echo(f"Coarsening by regridding to gcm grid...")
                variable_resolution = f"{variable_resolution}-coarsened-gcm"
                target_grid_filepath = os.path.join(os.path.dirname(__file__), '..', 'utils', 'target-grids', '60km', 'global', 'moose_grid.nc')
                ds = Regrid(target_grid_filepath, variables=job_spec["parameters"]["variables"], scheme="linear").run(ds)
            else:
                scale_factor = int(scale_factor)
                if scale_factor != 1:
                    typer.echo(f"Coarsening {scale_factor}x...")
                    variable_resolution = f"{variable_resolution}-coarsened-{scale_factor}x"
                    ds, orig_ds = Coarsen(scale_factor=scale_factor).run(ds)
        elif job_spec['action'] == "regrid":
            if target_resolution != variable_resolution:
                typer.echo(f"Regridding to target resolution...")
                target_grid_filepath = os.path.join(os.path.dirname(__file__), '..', 'utils', 'target-grids', target_resolution, 'uk', 'moose_pr_grid.nc')

                ds = Regrid(target_grid_filepath, variables=[config['variable']]).run(ds)
        elif job_spec['action'] == "vorticity":
            typer.echo(f"Computing vorticity...")
            ds = Vorticity().run(ds)
        elif job_spec['action'] == "select-subdomain":
            typer.echo(f"Select {domain.value} subdomain...")
            if target_resolution == "2.2km":
                size = 64
            elif target_resolution == "2.2km-coarsened-8x":
                size = 16
            else:
                size = 32
            ds = SelectDomain(subdomain=domain.value, size=size).run(ds)
        elif job_spec['action'] == "constrain":
            typer.echo(f"Filtering...")
            ds = Constrain(query=job_spec['query']).run(ds)
        else:
            raise f"Unknown action {job_spec['action']}"
    if domain == DomainOption.london and target_resolution == "2.2km":
        assert len(ds.grid_latitude) == 64
        assert len(ds.grid_longitude) == 64

    data_basedir = os.path.join(os.getenv("DERIVED_DATA"), "moose")

    output_metadata = UKCPDatasetMetadata(data_basedir, frequency=frequency, domain=domain.value, resolution=f"{variable_resolution}-{target_resolution}", ensemble_member='01', variable=config['variable'])

    logger.info(f"Saving data to {output_metadata.filepath(year)}")
    os.makedirs(output_metadata.dirpath(), exist_ok=True)
    ds.to_netcdf(output_metadata.filepath(year))
