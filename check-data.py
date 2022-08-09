import os

import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata

domain_var_resolutions = {
    "london": {
        "pr": [
            "2.2km-2.2km",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x"
        ],
        "vorticity850": [
            "2.2km-2.2km",
            "2.2km-coarsened-8x-2.2km",
            "2.2km-coarsened-gcm-2.2km",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x",
        ]
    },
    "birmingham": {
        "pr": [
            # "2.2km-2.2km",
            "2.2km-coarsened-4x-2.2km-coarsened-4x",
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x",
        ],
        "vorticity850": [
            # "2.2km-2.2km",
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x",
        ]
    }
}

years = list(range(1981, 2001))+list(range(2021, 2041))+list(range(2061, 2081))

for domain, var_resolutions in domain_var_resolutions.items():
    for var, resolutions in var_resolutions.items():
        for res in resolutions:
            print(f"Checking {var} over {domain} at {res}")

            bad_years = {"NaNs": set(), "no file": set(), "forecast_encoding": set(), "forecast_vars": set()}
            for year in years:
                var_meta = UKCPDatasetMetadata(os.getenv("MOOSE_DERIVED_DATA"), variable=var, frequency="day", domain=domain, resolution=res)

                try:
                    ds = xr.open_dataset(var_meta.filepath(year))
                except FileNotFoundError:
                    bad_years["no file"].add(year)
                    continue

                nan_count = ds[var].isnull().sum().values.item()

                if nan_count > 0:
                    bad_years["NaNs"].add(year)

                # check for forecast related metadata (should have been stripped)
                for v in ds.variables:
                    if "coordinates" in ds[v].encoding:
                        bad_years["forecast_encoding"].add(year)
                    if v in ["forecast_period", "forecast_reference_time", "realization", "forecast_period_bnds"]:
                       bad_years["forecast_vars"].add(year)

            # report findings
            for reason, error_years in bad_years.items():
                if len(error_years) > 0:
                    print(f"Failed '{reason}': {var} over {domain} at {res} for {error_years}")
