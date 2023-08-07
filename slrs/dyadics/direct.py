from dataclasses import dataclass, field
from datetime import datetime as dt
from tqdm.contrib.itertools import product
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

from slrs.utils.logging import logger

if TYPE_CHECKING:
    from slrs.types.calculation import Calculation

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import scipy.special as sp
from numba import jit, prange, complex128, float64, int64, set_num_threads

def gamma(t: npt.NDArray[np.float64]):
    t = np.asarray(t)
    mask = np.abs(t) >= 1.0
    out = np.zeros_like(t, dtype=np.complex128)
    np.sqrt(1 - t ** 2 + 0j, out=out, where=np.invert(mask))
    out = - 1j * out
    np.sqrt(t ** 2 - 1 + 0j, out=out, where=mask)
    return out

def single_point(
    scratch: npt.NDArray[np.complex128],
    wavevectors_im: npt.NDArray[np.float64],
    delta_point_m: npt.NDArray[np.float64],
    gf: npt.NDArray[np.complex128],
    prod: npt.NDArray[np.complex128],
):
    inv_delta_radius_m = np.eye(3) / np.linalg.norm(delta_point_m)
    delta_radius_m = np.eye(3) * np.linalg.norm(delta_point_m)

    rr = np.outer(delta_point_m, delta_point_m) * inv_delta_radius_m ** 2

    gf[...] = np.exp(
        1j * wavevectors_im[..., np.newaxis, np.newaxis] * delta_radius_m[np.newaxis, ...]
    ) / (4 * np.pi) * inv_delta_radius_m[np.newaxis, ...]

    prod[...] = inv_delta_radius_m[np.newaxis, ...] / wavevectors_im[..., np.newaxis, np.newaxis]

    scratch[...] = (
        np.eye(3)[np.newaxis, ...] + 1j * prod - 1.0 * prod ** 2
            + (
                -1.0 - 3j * prod
                + 3.0 * prod ** 2
            ) * rr[np.newaxis, ...]
    ) * gf


def loop_inner(
    out,
    wavevectors_im,
    num_sites,
    extent,
    basis_vectors,
    ):
    gf = np.zeros((wavevectors_im.size,3, 3), dtype=np.complex128)
    prod = np.zeros((wavevectors_im.size, 3, 3), dtype=np.complex128)
    scratch = np.zeros((wavevectors_im.size, 3, 3), dtype=np.complex128)
    for ii, jj in product(range(num_sites), range(num_sites), leave=False):
        if ii == jj:
            continue
        r_i = (
            ii % extent[0] * basis_vectors[0]
                + ii // extent[0] * basis_vectors[1]
        )
        r_j = (
            jj % extent[0] * basis_vectors[0]
                 + jj // extent[0] * basis_vectors[1]
        )
        delta_m = r_i - r_j

        scratch[:] = 0.0
        gf[:] = 0.0
        prod[:] = 0.0

        single_point(
            scratch,
            wavevectors_im,
            delta_m,
            gf,
            prod
        )
        out[..., 3 * ii : 3 * (ii + 1), 3 * jj : 3 * (jj + 1)] = scratch


class DirectDyadic:

    @staticmethod
    def _construct_inner(
        wavevectors_im,
        extent,
        basis_vectors,
    ):
        num_sites = extent[0] * extent[1]
        result = np.zeros((wavevectors_im.shape[0], wavevectors_im.shape[1], 3 * num_sites), dtype=np.complex128)
        unique_wavevectors_im = wavevectors_im[:, 0]

        out = np.zeros((unique_wavevectors_im.size, 3 * num_sites, 3 * num_sites), dtype=np.complex128)

        loop_inner(
            out,
            unique_wavevectors_im,
            num_sites,
            extent,
            basis_vectors,
        )

        return out


    def construct(
        self,
        extent: List[int],
        basis_vectors: List[npt.NDArray],
        coupled_mode_index: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        num_sites = extent[0] * extent[1]
        logger.info(f"constructing Green's dyadic for {num_sites} lattice nodes:")
        start = dt.now()

        wavevectors_im = 2 * np.pi / (1e-9 * wavelengths_nm) * coupled_mode_index

        out = self._construct_inner(
            wavevectors_im,
            np.asarray(extent),
            np.asarray(basis_vectors),
        )


        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"Green's dyadic constructed in  {seconds_elapsed}s")

        return out

