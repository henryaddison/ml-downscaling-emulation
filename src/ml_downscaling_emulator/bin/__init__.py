from pathlib import Path
from typing import List

import typer
import xarray as xr

from . import evaluate

app = typer.Typer()
app.add_typer(evaluate.app, name="evaluate")


@app.command()
def sample(files: List[Path]):
    for file in files:
        ds = xr.open_dataset(file)
        # take something from each season and each decade
        sampled_ds = ds.sel(
            time=((ds["time.month"] % 3 == 0) & (ds["time.year"] % 10 == 0))
        ).load()
        ds.close()
        del ds
        print(f"Saving {file}")
        sampled_ds.to_netcdf(file)
        del sampled_ds


if __name__ == "__main__":
    app()
