# Script to convert zipped data files to netcdf
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from time import time

import xarray as xr
from numpy import load as np_load
from pyPTerodaCTILES.io.batch_converter import batch_converter, batch_converter_zip
from pyPTerodaCTILES.io.readers import (
    ColumnDiags,
    ConvergenceyDiags,
    GlobalDiags,
    ICPLevels,
    ICWLevels,
    PTerodaCTILES_FileFormat,
    SliceDiags,
    StabilityDiags,
)


def global_converter(data_source: Path, output_path: Path) -> None:
    """convert global files; wrapper around batch converter dependent on zip vs directory source"""

    # Global diagnostics : collect in one file, may have missing components
    glob_readers = {
        "ICplevels": ICPLevels(),
        "ICwlevels": ICWLevels(),
        "globaldiags": GlobalDiags(),
        "stabilitydiags": StabilityDiags(),
        "solver_convergence": ConvergenceyDiags(),
    }
    print(f"{output_path.name}: Will convert IC and global diagnostics files")

    t0 = time()
    glob_xds = {}
    if data_source.is_dir():
        for pattern in glob_readers:
            try:
                glob_xds[pattern] = glob_readers[pattern].load(
                    list(data_source.glob(pattern))[0]
                )
            except IndexError:
                print(f"Skipping missing {pattern} input")
    elif data_source.suffix == ".zip":
        with np_load(data_source) as zipfile:
            for pattern in glob_readers:
                try:
                    glob_xds[pattern] = glob_readers[pattern].load(
                        [zipfile[x] for x in zipfile if pattern in x][0]
                    )
                except IndexError:
                    print(f"Skipping missing {pattern} input")

    ic_xds = [glob_xds.pop(k) for k in ("ICplevels", "ICwlevels") if k in glob_xds]
    if ic_xds:
        xr.merge(ic_xds).to_netcdf(
            output_path, engine="h5netcdf", mode="w", group="IC", compute=True
        )
    for group, xds in glob_xds.items():
        xds.to_netcdf(
            output_path, engine="h5netcdf", mode="a", group=group, compute=True
        )
    print(f"Total time         : {time() - t0}s")


def local_converter(
    data_source: Path,
    glob_pattern: str,
    reader: PTerodaCTILES_FileFormat,
    output_path: Path,
    tindex_slice: slice = slice(None),
    batch_size: int = 500,
    mode: str = "cheap",
) -> None:
    """wrapper around batch converter dependent on zip vs directory source
    'cheap' mode loads files in memory batch by batch
    'fast' mode loads all files in memory at once
    """

    # data source is a directory
    if data_source.is_dir():
        batch_converter(
            file_reader=reader,
            input_files=sorted(data_source.glob(glob_pattern))[tindex_slice],
            output_file_name=output_path,
            batch_size=batch_size,
        )
    # data source is a zip file  -- faster !!!
    elif data_source.suffix == ".zip":
        with np_load(data_source) as zipfile:
            # get filenames
            input = sorted([x for x in zipfile if glob_pattern.strip("*") in x])[
                tindex_slice
            ]
            if mode == "fast":
                # load all data
                input = [zipfile[x] for x in input]
                batch_converter(
                    file_reader=reader,
                    input_files=input,
                    output_file_name=output_path,
                    batch_size=None,
                )
            elif mode == "cheap":
                batch_converter_zip(
                    file_reader=reader,
                    zip_master=data_source,
                    input_files=input,
                    output_file_name=output_path,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(
                    f"Unrecognized mode {mode}. Choose between 'cheap' and 'fast'"
                )
    else:
        raise ValueError(
            f"Unsupported format {data_source.name}. Use a directory or a .zip file."
        )


def parser():
    parser = ArgumentParser(
        description="Convert PTerodaCTILESv0.3 files into a collection of NetCDF files",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_source",
        type=Path,
        help="location of input PTerodaCTILES .dat files, can be a directory or a zip file",
    )
    parser.add_argument(
        "output_path", type=Path, help="output directory, will create if missing"
    )

    parser.add_argument(
        "--runid",
        type=int,
        default=1,
        help="""run id to match against the filenames;
                relevant for slice, columns, global and stability;
                """,
    )

    parser.add_argument(
        "--slice_batch_size",
        type=int,
        default=1000,
        help="""how many slice files to process at once;
                will create a temporary .nc file for each batch before combining them,
                larger batch puts more pressure on memory but is faster;
                """,
    )

    parser.add_argument(
        "-d",
        "--diagnostics",
        type=str,
        nargs="*",
        default=["all"],
        help="""choose diagnostics to convert""",
        choices=["all", "global", "column", "xy_slice", "xz_slice"],
    )

    parser.add_argument(
        "--tinit",
        type=int,
        default=None,
        help="""initial time index, default to first available""",
    )

    parser.add_argument(
        "--tend",
        type=int,
        default=None,
        help="""final time index, default to last available;
                only relevant for column and slice diagnostics
            """,
    )

    parser.add_argument(
        "--tstride",
        type=int,
        default=1,
        help="number of steps between time indices to process",
    )
    return parser.parse_args()


def main():
    # parse command line parameters
    args = parser()
    print("\nInput parameters:")
    print(f"data_source : {args.data_source}")
    print(f"runid: {args.runid}")
    print(f"time_slice: slice({args.tinit}:{args.tend}:{args.tstride})")
    print(f"diagnostics: {args.diagnostics}")
    print(f"output_path : {args.output_path}")
    print(f"slice_batch_size : {args.slice_batch_size}")
    print("")

    # parse runid into string with expected formatting
    runid = f"run{int(args.runid):06d}"

    diagnostics = args.diagnostics
    if "all" in diagnostics:
        diagnostics = ["global", "column", "xy_slice", "xz_slice"]

    # make sure the output path exists
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Global diagnostics : collect in one file, may have missing components
    if "global" in diagnostics:
        global_converter(
            data_source=args.data_source,
            output_path=args.output_path / f"{runid}globaldiags.nc",
        )
        print("")

    # column diagnostics
    if "column" in diagnostics:
        local_converter(
            data_source=args.data_source,
            glob_pattern=f"{runid}columndiags*",
            reader=ColumnDiags(),
            output_path=args.output_path / f"{runid}columndiags.nc",
            tindex_slice=slice(args.tinit, args.tend, args.tstride),
            batch_size=None,
            mode="fast",
        )
        print("")

    # slice diagnostics
    for slice_ax in ["xy", "xz"]:
        if f"{slice_ax}_slice" in diagnostics:
            local_converter(
                data_source=args.data_source,
                glob_pattern=f"{runid}_{slice_ax}_diags*",
                reader=SliceDiags(slice_ax),
                output_path=args.output_path / f"{runid}_{slice_ax}_diag.nc",
                tindex_slice=slice(args.tinit, args.tend, args.tstride),
                batch_size=args.slice_batch_size,
                mode="cheap",
            )
            print("S")


if __name__ == "__main__":
    main()
