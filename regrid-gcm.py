import argparse
import glob
import itertools
import os

import cftime
import iris

from preprocessing.regrid import Regrid

variables = ["pr", "psl"]
# years = itertools.chain(range(1980, 2000), range(2020, 2040), range(2060, 2080))
years = range(1980, 1982)

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
    parser.add_argument('--input', dest='input_dir', type=str, required=True,
                        help='Path to directory holding raw files')
    parser.add_argument('--target', dest='target_file', type=str, required=True,
                        help='Base path to raw data storage')
    parser.add_argument('--output', dest='output_dir', type=str, required=True,
                        help='Path to directory to store regridded files')
    parser.add_argument('--scheme', dest='regrid_scheme', type=str, required=False,
                        choices=Regrid.SCHEMES.keys(), default='linear',
                        help='Regridding scheme')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outputs = Regrid(args.input_dir, args.target_file, args.output_dir, args.regrid_scheme).run()

    print(outputs)