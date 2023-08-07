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

@jit(
    (complex128[:, :, :], float64[:], float64[:], complex128[:, :, :], complex128[:, :, :]),
    cache=True,
    nopython=True
)
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

    prod[...] = delta_radius_m[np.newaxis, ...] / wavevectors_im[..., np.newaxis, np.newaxis]

    scratch[...] = (
        np.eye(3)[np.newaxis, ...] + 1j * prod - 1.0 * prod ** 2
            + (
                -1.0 - 3j * prod
                + 3.0 * prod ** 2
            ) * rr[np.newaxis, ...]
    ) * gf


@jit(
    (complex128[:, :, :], float64[:], int64, int64[:], float64[:, :]), nopython=True
)
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
    for ii in range(num_sites):
        for jj in range(num_sites):
            if ii == jj:
                continue
            delta_m = (
                (ii - jj) % extent[0] * basis_vectors[0]
                    + (ii - jj) // extent[0] * basis_vectors[1]
            )

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
    @jit(
        complex128[:, :, :](complex128[:, :, :], complex128[:, :, :], float64, float64[:, :], int64[:], float64[:, :]),
        nopython=True,
        parallel=True
    )
    def _construct_inner(
        lattice_inverse_polarisability,
        source_vector,
        background_index,
        wavevectors_im,
        extent,
        basis_vectors,
    ):
        num_sites = extent[0] * extent[1]
        result = np.zeros((wavevectors_im.shape[0], wavevectors_im.shape[1], 3 * num_sites), dtype=np.complex128)


        # for ii in range(num_sites):
        #     for jj in range(num_sites):
        #for (kk, row) in enumerate(wavevectors_im):
        for kk in prange(wavevectors_im.shape[0]):
            print(f"on iter {kk} of {wavevectors_im.shape[0]}")
            row = wavevectors_im[kk]
            out = np.zeros((row.size, 3 * num_sites, 3 * num_sites), dtype=np.complex128)
            loop_inner(
                out,
                row,
                num_sites,
                extent,
                basis_vectors,
            )

            for (ll, wavevector_im) in enumerate(row):
                result[kk, ll] = np.linalg.solve(
                    lattice_inverse_polarisability[kk] - out[ll] * wavevector_im ** 2 / background_index ** 2,
                    source_vector[kk, ll, ...]
                )

        return result


    def construct(
        self,
        lattice_inverse_polarizability,
        source_vector,
        background_index,
        extent: List[int],
        basis_vectors: List[npt.NDArray],
        wavelengths_nm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        num_sites = extent[0] * extent[1]
        logger.info(f"constructing Green's dyadic for {num_sites} lattice nodes:")
        start = dt.now()

        wavevectors_im = 2 * np.pi / (1e-9 * wavelengths_nm)

        num_threads = 4
        set_num_threads(num_threads)
        logger.info(f"running on {num_threads} threads")


        out = self._construct_inner(
            lattice_inverse_polarizability,
            source_vector,
            background_index,
            wavevectors_im,
            np.asarray(extent),
            np.asarray(basis_vectors),
        )


        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"Green's dyadic constructed in  {seconds_elapsed}s")

        return out

