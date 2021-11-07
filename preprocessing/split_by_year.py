import itertools

import cftime
import xarray

"""
Split up an nc multifile dataset into single file per year.
"""
class SplitByYear:

    def __init__(self, input_dir, output_filepath_prefix, years = itertools.chain(range(1980, 2000), range(2020, 2040), range(2060, 2080))) -> None:
        self.input_dir = input_dir
        self.output_filepath_prefix = output_filepath_prefix
        self.years = years

        pass

    def run(self):
        output_files = []

        input = xarray.open_mfdataset(str(self.input_dir/"*.nc"))


        for year in self.years:
            single_year_input = input.sel(time=slice(cftime.Datetime360Day(year, 12, 1, 12, 0, 0, 0) , cftime.Datetime360Day(year+1, 11, 30, 12, 0, 0, 0)))

            output_filepath = f"{self.output_filepath_prefix}_{year}1201-{year+1}1130.nc"
            single_year_input.to_netcdf(output_filepath)
            output_files.append(output_filepath)

        print("All done")

        return output_files
