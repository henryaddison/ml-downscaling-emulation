import argparse
import glob
import itertools
import os

import cftime
import iris

variables = ["pr", "psl"]

def gcm_file(year, var):
    if (year % 10) <= 8:
        start = (year // 10) * 10 - 1
    else:
        start = year

    end = start + 10

    return f"{args.data_dir}/60km/rcp85/01/{var}/day/{var}_rcp85_land-gcm_uk_60km_01_day_{start}1201-{end}1130.nc"

def get_args():
    parser = argparse.ArgumentParser(description='Regrid GCM data to match the CPM data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', dest='data_dir', type=str, required=True,
                        help='Base path to raw data storage')
    parser.add_argument('--derived', dest='derived_data', type=str, required=True,
                        help='Base path to storage for derived data')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    cpm_files = glob.glob(f"{args.data_dir}/2.2km/rcp85/01/pr/day/*.nc")

    cpm_cube = iris.load_cube(cpm_files[0])

    for var in variables:
        gcm_files = f"{args.data_dir}/60km/rcp85/01/{var}/day/*19*.nc"

        output_dir = f"{args.derived_data}/60km-2.2km-regrid-lin/rcp85/01/{var}/day"
        os.makedirs(output_dir, exist_ok=True)

        for year in  itertools.chain(range(1980, 2000), range(2020, 2040), range(2060, 2080)):
            gcm_cube = iris.load_cube(gcm_file(year, var))

            single_year_gcmcube = gcm_cube.extract(iris.Constraint(time=lambda cell: cftime.Datetime360Day(year, 12, 1, 12, 0, 0, 0) <= cell.point <= cftime.Datetime360Day(year+1, 11, 30, 12, 0, 0, 0)))

            rp_gcm_cube = single_year_gcmcube.regrid(cpm_cube, iris.analysis.Linear())

            iris.save(rp_gcm_cube, f"{output_dir}/{var}_rcp85_land-gcm_uk_60km_01_day_rp_regrid_{year}1201-{year+1}1130.nc")

    print("All done")