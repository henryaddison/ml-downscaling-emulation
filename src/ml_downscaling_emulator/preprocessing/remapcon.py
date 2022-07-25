import tempfile
from cdo import Cdo

class Remapcon:
    def __init__(self, target_grid_filepath):
        self.target_grid_filepath = target_grid_filepath

    def run(self, ds):
        input_file = tempfile.NamedTemporaryFile(delete=True, prefix='cdo_xr_input_', dir=tempfile.gettempdir())
        print(input_file.name)
        ds.to_netcdf(input_file.name)

        cdo = Cdo()
        ds = cdo.remapcon(self.target_grid_filepath, input=input_file.name, returnXDataset = True)

        if "latitude_longitude" in ds.variables:
            lat_name = 'latitude'
            lon_name = 'longitude'
        elif "rotated_latitude_longitude" in ds.variables:
            lat_name = 'grid_latitude'
            lon_name = 'grid_longitude'
        else:
            raise RuntimeError("Unrecognised grid system")

        ds[lat_name] = ds[lat_name].assign_attrs(standard_name=lat_name)
        ds[lon_name] = ds[lon_name].assign_attrs(standard_name=lon_name)

        return ds