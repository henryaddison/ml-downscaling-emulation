import glob
import itertools
import os

import cftime
import iris

"""
Regrid all nc files in input directory based on the given target grid file and save output

Assumes input files cover 10 years but output should cover 1 year.
"""
class Regrid:

    SCHEMES = {
        "linear": iris.analysis.Linear,
        "nn": iris.analysis.Nearest,
        "area-weighted": iris.analysis.AreaWeighted
    }

    def __init__(self, input_dir, target_grid, output_dir, scheme) -> None:
        self.input_dir = input_dir
        self.target_grid = target_grid
        self.output_dir = output_dir
        self.scheme = self.SCHEMES[scheme]()

        self.years = itertools.chain(range(1980, 2000), range(2020, 2040), range(2060, 2080))
        self.years = range(1980, 1989)
        pass

    def run(self):
        output_files = []

        target_cube = iris.load_cube(self.target_grid)

        example_input_cube_file = glob.glob(f"{self.input_dir}/*.nc")[0]
        example_input_cube = iris.load_cube(example_input_cube_file)

        regridder = self.scheme.regridder(example_input_cube, target_cube)

        for year in self.years:
            input_fileglob = f"{self.input_dir}*_{self.input_file_year_range(year)}.nc"
            input_filepath = glob.glob(input_fileglob)[0]
            input_cube = iris.load_cube(input_filepath)

            single_year_input_cube = input_cube.extract(iris.Constraint(time=lambda cell: cftime.Datetime360Day(year, 12, 1, 12, 0, 0, 0) <= cell.point <= cftime.Datetime360Day(year+1, 11, 30, 12, 0, 0, 0)))

            single_year_regridded_cube =  regridder(single_year_input_cube)

            output_filename = f"{os.path.basename(input_filepath)[0:-20]}{year}1201-{year+1}1130.nc"
            output_filepath = f"{self.output_dir}/{output_filename}"
            iris.save(single_year_regridded_cube, output_filepath)
            output_files.append(output_filepath)

        print("All done")

        return output_files


    def input_file_year_range(self, year):
        if (year % 10) <= 8:
            start = (year // 10) * 10 - 1
        else:
            start = year

        end = start + 10

        return f"{start}1201-{end}1130"