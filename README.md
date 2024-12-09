# Python tools and utilities for the PTerodaCTILES fluids code

## Install
Can install as a python package with

 * download the repository
 * run (preferably within a virtual environment)
   ```pip install --editable <location-of-repository>[dev]```
   This should install any missing dependencies as well.
The package is in a development stage, hence the editable install.


## Contains
 * Python IO readers for the ```.dat``` diagnostic files produced by PTerodaCTILESv0.3. Expose them by importing ```pyPTerodaCTILES.IO.readers``` . The output is in self-descriptive xarray datasets.
 * Python conversion CLI to convert them to NetCDF format, e.g. a command line tool `convert_pterodactiles_dat_to_netcdf` is exposed on installation.


