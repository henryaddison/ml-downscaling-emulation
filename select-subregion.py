import argparse
import glob
import logging
import os
from pathlib import Path

import xarray as xr

def get_args():
    parser = argparse.ArgumentParser(description='Select a sub-region of a UKCP18 file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input_dir', type=Path, required=True,
                        help='Base path to input storage')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Base path to storage output')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    full_da_files = glob.glob(str(args.input_dir / "*.nc"))

    for full_file in full_da_files:
        logging.info(f"Working on {full_file}")
        da = xr.open_dataset(full_file)

        output_file = args.output_dir / f"london_{os.path.basename(full_file)}"

        subregion_da = da.isel(grid_latitude=range(139,199), grid_longitude=range(331, 391))
        subregion_da.to_netcdf(output_file)

    logging.info("All done")