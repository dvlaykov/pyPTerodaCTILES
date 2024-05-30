# readers for parsing diagnostic output files for PTerodaCTILES_v0p3

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO  # StringIO behaves like a file object -- for zipped dirs
from pathlib import Path

# class tools
from typing import ClassVar, Union

import xarray as xr
from numpy import array as np_array
from numpy import ceil
from numpy import empty as np_empty
from numpy import loadtxt as np_loadtxt
from numpy import ndarray


# PTerodaCTILES output file name formatss
def diagnostics_filename(runid: int, type: str, time: Union[int, None] = None) -> str:
    if type in ["ICplevels", "ICwlevels", "solver_convergence"]:
        return f"{type}.dat"
    elif type in ["globaldiags", "stabilitydiags"]:
        return f"run{runid:06d}{type}.dat"
    elif type in ["_xy_diags_", "_xz_diags_", "_yz_diags_", "columndiags_"]:
        assert time is not None
        return f"run{runid:06d}{type}{time:010d}.dat"
    else:
        raise ValueError(f"unrecognised file type: {type}")


@dataclass(frozen=True)
class PTerodaCTILES_FileFormat:
    """Abstract File reader"""

    def load(self, file: Union[Path, str, bytes]) -> xr.Dataset:
        """file can be a filename or a bytes object,
        e.g. output from a np.load() acting on a zip file
        """
        # try to treat file as a path name in the file system
        if isinstance(file, (Path, str)):
            arr = np_loadtxt(file)
        # this is useful for accessing files inside a zipped folder e.g. opened with np.load
        elif isinstance(file, bytes):
            arr = np_loadtxt(StringIO(file.decode()))
        else:
            raise TypeError(f"unsupported file type {type(file)}")
        return self._parse_into_xds_(arr)

    @abstractmethod
    def _parse_into_xds_(self, array: ndarray) -> xr.Dataset:
        """parse numpy array as an xarray with implied metadata"""
        ...


@dataclass(frozen=True)
class GlobalDiags(PTerodaCTILES_FileFormat):
    """Format reader for global diagnostics file"""

    _vars_: ClassVar[Iterable[str]] = [
        "time",
        "mass",
        "umom",
        "vmom",
        "pe",
        "ke",
        "tke",
        "massw",
        "water",
        "vapour",
        "liquid",
        "frozen",
        "entropy",
        "ie",
        "sfc_src_mass",
        "sfc_src_water",
        "sfc_src_entropy",
        "sfc_src_energy",
        "sfc_src_umom",
        "sfc_src_vmom",
        "lrg_scl_src_mass",
        "lrg_scl_src_water",
        "lrg_scl_src_entropy",
        "lrg_scl_src_energy",
        "lrg_scl_src_umom",
        "lrg_scl_src_vmom",
        "accum_pe_dissn",
        "accum_ke_dissn",
        "accum_ie_dissn",
    ]
    _dims_: ClassVar[Iterable[str]] = ["time"]

    def _parse_into_xds_(self, array: ndarray) -> xr.Dataset:
        vars = {}
        for i, v in enumerate(self._vars_):
            vars[v] = array[:, i]
        data_vars = {
            d: xr.DataArray(vars[d], dims=self._dims_)
            for d in vars
            if d not in self._dims_
        }
        coords = {d: vars[d] for d in self._dims_}
        return xr.Dataset(data_vars, coords)


@dataclass(frozen=True)
class ConvergenceyDiags(GlobalDiags):
    """Format reader for convergence diagnostics output"""

    _vars_: ClassVar = [
        "step",
        "iter",
        "res_rho",
        "res_qt",
        "res_etat",
        "res_u",
        "res_v",
        "res_w_dp",
        "res_buoy",
        "rhs_help",
        "p_inc",
    ]
    _dims_: ClassVar[Iterable[str]] = ["iter", "step"]

    def _parse_into_xds_(self, array: ndarray) -> xr.Dataset:
        vars = {}
        for i, v in enumerate(self._vars_):
            vars[v] = array[:, i]
        niter = max(vars["iter"])
        nstep = max(vars["step"])
        data_vars = {
            d: xr.DataArray(vars[d].reshape([niter, nstep]), dims=self._dims_)
            for d in vars
            if d not in self._dims_
        }
        coords = {d: vars[d] for d in self._dims_}
        return xr.Dataset(data_vars, coords)


@dataclass(frozen=True)
class ICPLevels(GlobalDiags):
    """Format reader for IC p-levels output"""

    _vars_: ClassVar = ["zp", "rho", "p", "u", "v"]
    _dims_: ClassVar[Iterable[str]] = ["zp"]


@dataclass(frozen=True)
class ICWLevels(GlobalDiags):
    """Format reader for IC w-levels output"""

    _vars_: ClassVar = ["zw", "qt", "qv", "ql", "qf", "rh", "T", "theta", "thetal"]
    _dims_: ClassVar[Iterable[str]] = ["zw"]


@dataclass(frozen=True)
class StabilityDiags(GlobalDiags):
    """Format reader for stability diagnostics output"""

    _vars_: ClassVar = {
        "time": 0,
        # Acoustic wave Courant numbers
        "CFL_acoustic": [1, 2, 3],
        # Gravity wave Courant number  sqrt(N^2) * dt
        "CFL_gravity": 4,
        # Convective Courant number  sqrt(-N^2) * dt
        "CFL_convect": 5,
        # Advective Courant numbers
        "CFL_advect": [6, 7, 8],
        # Deformational Courant numbers ~ grad u * dt
        "CFL_deform": slice(9, 18),
    }
    _dims_: ClassVar[Iterable[str]] = ["time", "vec1", "vec2"]

    def _parse_into_xds_(self, array: ndarray) -> xr.Dataset:
        vars = {}
        vars["CFL_acoustic"] = xr.DataArray(
            array[:, self._vars_["CFL_acoustic"]], dims=["time", "vec1"]
        )
        vars["CFL_gravity"] = xr.DataArray(
            array[:, self._vars_["CFL_gravity"]], dims=["time"]
        )
        vars["CFL_convect"] = xr.DataArray(
            array[:, self._vars_["CFL_convect"]], dims=["time"]
        )
        vars["CFL_advect"] = xr.DataArray(
            array[:, self._vars_["CFL_advect"]], dims=["time", "vec1"]
        )
        vars["CFL_deform"] = xr.DataArray(
            array[:, self._vars_["CFL_deform"]].reshape(array.shape[0], 3, 3),
            dims=["time", "vec1", "vec2"],
        )

        coords = {
            "time": array[:, self._vars_["time"]],
            "vec1": np_array(["x", "y", "z"]),
            "vec2": np_array(["dx", "dy", "dz"]),
        }
        return xr.Dataset(vars, coords)


@dataclass(frozen=True)
class ColumnDiags(PTerodaCTILES_FileFormat):
    """Format reader for column diagnostics file (at one time)"""

    _vars_: ClassVar[Iterable[str]] = [
        "zp",
        "zw",
        "p",
        "u",
        "v",
        "w_at_p",
        "ke",
        "uvar",
        "vvar",
        "wvar",
        "tke",
        "uflx",
        "vflx",
        "w",
        "water",
        "vapour",
        "liquid",
        "frozen",
        "etat",
        "qvar",
        "qflx",
        "etaflx",
        "temperature",
        "p_at_w",
        "rh",
        "n2",
        "alpha",
        "cld_frac",
        "rad_trcr",
        "rad_trcr_var",
        "updraft_indi",
        "entrainment",
        "detrainment",
        "pe_dissipation",
        "ke_dissipation",
        "ie_dissipation",
    ]

    _short_vars_: ClassVar[Iterable[str]] = [
        "zp",
        "p",
        "u",
        "v",
        "w_at_p",
        "ke",
        "uvar",
        "vvar",
        "wvar",
        "tke",
        "uflx",
        "vflx",
        "rad_trcr",
        "rad_trcr_var",
        "pe_dissipation",
        "ke_dissipation",
    ]

    def _parse_into_xds_(self, array: ndarray) -> xr.Dataset:
        array_seq = array.ravel()
        nz = int(array_seq[1])
        data_vars = {"time": array_seq[:1]}
        idx = 2
        for d in self._vars_:
            # these vars are nz long
            if d in self._short_vars_:
                data_vars[d] = array_seq[idx : idx + nz]
                idx += nz
            # these vars are (nz + 1) long
            else:
                data_vars[d] = array_seq[idx : idx + (nz + 1)]
                idx += nz + 1
        dims = {"time", "zp", "zw"}
        # upgrade fields to data arrays
        for d in self._vars_:
            if d not in dims:
                if d in self._short_vars_:
                    data_vars[d] = xr.DataArray(
                        data_vars[d][None, :], dims=["time", "zp"]
                    )
                else:
                    data_vars[d] = xr.DataArray(
                        data_vars[d][None, :], dims=["time", "zw"]
                    )

        return xr.Dataset(
            data_vars={d: data_vars[d] for d in data_vars if d not in dims},
            coords={d: data_vars[d] for d in dims},
        )


# file reader
@dataclass(frozen=True)
class SliceDiags(PTerodaCTILES_FileFormat):
    """Format reader for slice diagnostics file"""

    slice_type: str

    _nlist_max: ClassVar = 32
    _vars_: ClassVar = [
        "u",
        "v",
        "w",
        "p",
        "rho",
        "T",
        "qt",
        "qv",
        "ql",
        "qf",
        "rh",
        "etat",
        "etad",
        "etav",
        "etal",
        "etaf",
        "ctop",
        "trcp",
        "trcw",
        "entt",
        "diss",
    ]

    def _slice_dims_(self, slice_type: str, d: int):
        """pick dimensions based on slice and variable"""

        if slice_type == "xy":
            if d == "u":
                dims = [f"{slice_type[0]}_v", f"{slice_type[1]}_p"]
            elif d == "v":
                dims = [f"{slice_type[0]}_p", f"{slice_type[1]}_v"]
            else:
                dims = [f"{slice_type[0]}_p", f"{slice_type[1]}_p"]
        elif slice_type == "xz":
            if d == "u":
                dims = [f"{slice_type[0]}_v", f"{slice_type[1]}_p"]
            elif d in ["v", "p", "rho", "diss", "trcp"]:
                dims = [f"{slice_type[0]}_p", f"{slice_type[1]}_p"]
            else:
                dims = [f"{slice_type[0]}_p", f"{slice_type[1]}_v"]
        elif slice_type == "yz":
            if d == "v":
                dims = [f"{slice_type[0]}_v", f"{slice_type[1]}_p"]
            elif d in ["u", "p", "rho", "diss", "trcp"]:
                dims = [f"{slice_type[0]}_p", f"{slice_type[1]}_p"]
            else:
                dims = [f"{slice_type[0]}_p", f"{slice_type[1]}_v"]
        else:
            raise ValueError("Unsupported slice type")

        return dims

    def _parse_into_xds_(self, array: ndarray) -> xr.Dataset:
        array_seq = array.ravel()
        max_idx = len(array_seq) - 1

        assert self.slice_type in ["xy", "yz", "xz"]
        slice_type = self.slice_type

        coords = {"time": array_seq[:1]}
        # Now some sequential reading

        # various field dimensions
        n1, n2p, nlev, ntracerp, ntracerw = array_seq[1:6].astype(int)
        # print (f'n1 = {n1}')
        # print (f'n2p = {n2p}')
        # print (f'nlev = {nlev}')
        # print (f'ntracerp = {ntracerp}')
        # print (f'ntracerw = {ntracerw}')
        if slice_type == "xy":
            n2w = n2p
        elif slice_type in ["xz", "yz"]:
            n2w = n2p + 1

        # field coordinates
        idx = 6
        coords[f"{slice_type[0]}_v"] = array_seq[idx : idx + n1]
        idx += n1
        coords[f"{slice_type[0]}_p"] = array_seq[idx : idx + n1]
        idx += n1
        coords[f"{slice_type[1]}_v"] = array_seq[idx : idx + n2w]
        idx += n2w
        coords[f"{slice_type[1]}_p"] = array_seq[idx : idx + n2p]
        idx += n2p
        level_dim = f"levels_{'xyz'.strip(slice_type)}"
        coords[level_dim] = array_seq[idx : idx + nlev]
        idx += nlev

        # print ('coords', coords.keys())
        # TODO: can do this better if we new how many datapoints here
        # discard junk padding
        while array_seq[idx] < 0:
            idx += 1
        # print ('after coords', idx, array_seq[idx])
        # assert idx < max_idx

        # Parse stored fields list
        # there should be at least one zero terminating the list.
        # Fields list code  "-1" because of Fortran 1-based indexing
        field_list = []
        while array_seq[idx] > 0:
            try:
                field_list.append(self._vars_[int(array_seq[idx]) - 1])
            except IndexError:
                print(
                    f"IndexError: list index out of range for var code {array_seq[idx]}"
                )
            idx += 1
        # print ('after list codes', idx, array_seq[idx])
        # assert idx < max_idx

        # Skip dummy entries in list of fields
        idx += self._nlist_max - len(field_list)
        # print ('after list codes skip', idx, array_seq[idx])
        # assert idx < max_idx

        # And skip any remaining padding zeros
        idx = 16 * int(ceil(idx / 16))
        # print ('after list codes 2nd skip', idx, array_seq[idx])
        # assert idx < max_idx
        # assert array_seq[idx-1] == 0

        # Size of one level of one data field
        # plus zero padding to 16
        field_sizep = 16 * int(ceil(n1 * n2p / 16))
        field_sizew = 16 * int(ceil(n1 * n2w / 16))
        # print (f'field sizes with padding: {field_sizep}, {field_sizew}')

        # Actually read fields
        # Can optimize the line-by-line reading
        # print ("Field_list:", field_list)
        data_vars = {}
        for i, f in enumerate(field_list):
            assert idx < max_idx
            dims = self._slice_dims_(slice_type, f)
            # Cloud top height has a single level
            if f == "ctop":
                data_vars[f] = array_seq[idx : idx + n1 * n2p].reshape(n2p, n1).T
                idx += field_sizep
            # ntracerp tracers on nlev levels
            elif f == "trcp":
                data_vars[f] = np_empty([n1, n2p, nlev, ntracerp])
                dims += [level_dim, "ptracer"]
                for itracer in range(ntracerp):
                    for ilev in range(nlev):
                        data_vars[f][:, :, ilev, itracer] = (
                            array_seq[idx : idx + n1 * n2p].reshape(n2p, n1).T
                        )
                        idx += field_sizep
            # ntracerp tracers on nlev levels
            elif f == "trcw":
                data_vars[f] = np_empty([n1, n2w, nlev, ntracerw])
                dims += [level_dim, "wtracer"]
                for itracer in range(ntracerw):
                    for ilev in range(nlev):
                        data_vars[f][:, :, ilev, itracer] = (
                            array_seq[idx : idx + n1 * n2w].reshape(n2w, n1).T
                        )
                        idx += field_sizew
            # one field on nlev levels on p-grid
            elif i in ["u", "v", "p", "rho", "diis"]:
                data_vars[f] = np_empty([n1, n2p, nlev])
                dims += [level_dim]
                for ilev in range(nlev):
                    data_vars[f][:, :, ilev] = (
                        array_seq[idx : idx + n1 * n2p].reshape(n2p, n1).T
                    )
                    idx += field_sizep
            # one field on nlev levels on w-grid
            else:
                data_vars[f] = np_empty([n1, n2w, nlev])
                dims += [level_dim]
                for ilev in range(nlev):
                    data_vars[f][:, :, ilev] = (
                        array_seq[idx : idx + n1 * n2w].reshape(n2w, n1).T
                    )
                    idx += field_sizew
            # convert array into xarray
            data_vars[f] = xr.DataArray(data_vars[f][None, ...], dims=["time"] + dims)

        return xr.Dataset(data_vars, coords)
