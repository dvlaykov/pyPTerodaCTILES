from collections.abc import Iterable
from pathlib import Path
from time import time
from typing import Union

import xarray as xr
from dask.diagnostics import ProgressBar
from numpy import ceil, load

from .readers import PTerodaCTILES_FileFormat


def batch_converter(
    file_reader: PTerodaCTILES_FileFormat,
    input_files: list[Path | str | bytes],
    output_file_name: Union[Path, str],
    batch_size: int = 1000,
) -> None:
    t0 = time()
    if batch_size is None:
        batch_size = len(input_files)
    nbatch = int(ceil(len(input_files) / batch_size))
    # convert filename to Path for ease
    out_path = Path(output_file_name)
    print(
        f"{out_path.name}: Will convert {len(input_files)} files in {nbatch} batch(es)"
    )
    batch_files = []
    for ibatch in range(nbatch):
        t = time()
        # get batch files
        imin = ibatch * batch_size
        imax = min((ibatch + 1) * batch_size, len(input_files))
        batch = input_files[imin:imax]

        # read files in batch
        batch_xds = xr.concat([file_reader.load(f) for f in batch], dim="time")
        # write to temporary nc batch file
        batch_files.append(out_path.with_stem(f"{out_path.stem}_{ibatch}"))
        batch_xds.to_netcdf(
            batch_files[ibatch], engine="h5netcdf", mode="w", compute=True
        )
        print(f"batch {ibatch}: {time() - t}s")
        del batch_xds

    # combine batch files
    if nbatch == 1:
        batch_files[0].rename(out_path)
    else:
        with ProgressBar():
            batch_xds = xr.open_mfdataset(batch_files, parallel=True)
            batch_xds.to_netcdf(out_path, mode="w", compute=True)
            batch_xds.close()
        # remove batch files
        for f in batch_files:
            f.unlink()
    print(f"Total time         : {time() - t0}s")


def batch_converter_zip(
    file_reader: PTerodaCTILES_FileFormat,
    zip_master: Union[Path | str],
    input_files: list[Path | str | bytes],
    output_file_name: Union[Path, str],
    batch_size: int = 1000,
) -> None:
    t0 = time()
    if batch_size is None:
        batch_size = len(input_files)
    nbatch = int(ceil(len(input_files) / batch_size))
    # convert filename to Path for ease
    out_path = Path(output_file_name)
    print(
        f"{out_path.name}: Will convert {len(input_files)} files in {nbatch} batch(es)"
    )

    batch_files = []
    with load(zip_master) as zipfile:
        for ibatch in range(nbatch):
            t = time()
            # get batch files
            imin = ibatch * batch_size
            imax = min((ibatch + 1) * batch_size, len(input_files))
            batch = input_files[imin:imax]

            # read files in batch
            batch_xds = xr.concat(
                [file_reader.load(zipfile[f]) for f in batch], dim="time"
            )
            # write to temporary nc batch file
            batch_files.append(out_path.with_stem(f"{out_path.stem}_{ibatch}"))
            batch_xds.to_netcdf(
                batch_files[ibatch], engine="h5netcdf", mode="w", compute=True
            )
            print(f"batch {ibatch}: {time() - t}s")
            del batch_xds

    # combine batch files
    if nbatch == 1:
        batch_files[0].rename(out_path)
    else:
        with ProgressBar():
            batch_xds = xr.open_mfdataset(batch_files, parallel=True)
            batch_xds.to_netcdf(out_path, mode="w", compute=True)
            batch_xds.close()
        # remove batch files
        for f in batch_files:
            f.unlink()
    print(f"Total time         : {time() - t0}s")
