import glob
import logging
import os

import xarray as xr

logger = logging.getLogger(__name__)

class SelectSubregion:
    def __init__(self, input_dir, output_dir, x_range=range(331, 391), y_range=range(139,199), x_dim="grid_longitude", y_dim="grid_latitude") -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subregion_defn = {x_dim: x_range, y_dim: y_range}

    def run(self):
        output_files = []
        full_da_files = glob.glob(str(self.input_dir / "*.nc"))

        for full_file in full_da_files:
            logger.info(f"Working on {full_file}")
            da = xr.open_dataset(full_file)

            output_filepath = self.output_dir / f"{os.path.basename(full_file)}"

            subregion_da = da.isel(self.subregion_defn)
            subregion_da.to_netcdf(output_filepath)
            output_files.append(output_filepath)

        logger.info("All done")

        return output_files