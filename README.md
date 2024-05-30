# Python tools and itilities for the PTerodaCTILES fluids code

## Install
Can install as a python package with 

 * download the repository
 * run (preferably within a virtual environment)
   ```pip install --editable <location-of-repository>```
   It should all install any missing dependencies as well.
The package is in a development stage, hence the editable install.
## Contains 
 * Python IO readers for the ```.dat``` diagnostic files produced by PTerodaCTILES. Expose by importing ```pyPTerodaCTILES.io.readers``` . The data is self-descriptive xarray datasets.
 * Python conversion tools to convert data to NetCDF format, e.g. a command line tool `convert_pterodactiles_dat_to_netcdf` is exposed on installation.
