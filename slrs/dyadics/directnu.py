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
from numba import jit

def gamma(t: npt.NDArray[np.float64]):
    t = np.asarray(t)
    mask = np.abs(t) >= 1.0
    out = np.zeros_like(t, dtype=np.complex128)
    np.sqrt(1 - t ** 2 + 0j, out=out, where=np.invert(mask))
    out = - 1j * out
    np.sqrt(t ** 2 - 1 + 0j, out=out, where=mask)
    return out

@jit(nopython=True)
def single_point(
    scratch: npt.NDArray[np.complex128],
    wavevectors_im: npt.NDArray[np.float64],
    delta_point_m: npt.NDArray[np.float64],
    inv_delta_radius_m,
    delta_radius_m,
    rr,
    gf,
    prod
):
    inv_delta_radius_m[...] = np.eye(3) / np.linalg.norm(delta_point_m)
    delta_radius_m[...] = np.eye(3) * np.linalg.norm(delta_point_m)

    rr[...] = np.outer(delta_point_m, delta_point_m) * inv_delta_radius_m ** 2

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


class DirectDyadic:

    def __init__(self):
        self.inv_delta_radius_m = np.zeros((3, 3), dtype=np.complex128)
        self.delta_radius_m = np.zeros((3, 3), dtype=np.complex128)
        self.rr = np.zeros((3, 3), dtype=np.complex128)


    @staticmethod
    # @jit(nopython=True)
    def _construct_inner(
        lattice_polarisability,
        source_vector,
        background_index,
        wavevectors_im,
        extent,
        basis_vectors,
        inv_delta_radius_m,
        delta_radius_m,
        rr,
        gf,
        prod
    ):
        num_sites = extent[0] * extent[1]
        out = np.zeros((wavevectors_im.shape[1], 3 * num_sites, 3 * num_sites), dtype=np.complex128)
        scratch = np.zeros((wavevectors_im.shape[1], 3, 3), dtype=np.complex128)

        result = np.zeros((*wavevectors_im.shape, 3 * num_sites), dtype=np.complex128)

        delta_m = np.zeros(3)

        # for ii in range(num_sites):
        #     for jj in range(num_sites):
        for kk in range(wavevectors_im.shape[0]):
            for ii in range(num_sites):
                for jj in range(num_sites):

                    delta_m[...] = (
                        (ii - jj) % extent[0] * basis_vectors[0]
                            + (ii - jj) // extent[0] * basis_vectors[1]
                    )

                    if ii == jj:
                        continue

                    single_point(
                        scratch,
                        wavevectors_im[kk],
                        delta_m,
                        inv_delta_radius_m,
                        delta_radius_m,
                        rr,
                        gf,
                        prod
                    )
                    out[..., 3 * ii : 3 * (ii + 1), 3 * jj : 3 * (jj + 1)] = scratch

                # for ii in range(wavevectors_im.shape[1]):
                # result[kk, ii] = np.linalg.solve(out[ii], source_vector[kk,ii, ...])
            result[kk] = np.linalg.solve(
                lattice_polarisability - 4 * np.pi * out * wavevectors_im[kk, ..., np.newaxis, np.newaxis] ** 2 / background_index ** 2,
                source_vector[kk, ...]
            )

        return result


    def construct(
        self,
        lattice_polarizability,
        source_vector,
        background_index,
        extent: List[int],
        basis_vectors: List[npt.NDArray],
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        num_sites = extent[0] * extent[1]
        logger.info(f"constructing Green's dyadic for {num_sites} lattice nodes:")
        start = dt.now()

        wavevectors_im = 2 * np.pi / (1e-9 * wavelengths_nm)

        logger.info("running")
        self.gf = np.zeros((wavelengths_nm.shape[1],3, 3), dtype=np.complex128)
        self.prod = np.zeros((wavelengths_nm.shape[1], 3, 3), dtype=np.complex128)

        out = self._construct_inner(
            lattice_polarizability,
            source_vector,
            background_index,
            wavevectors_im,
            np.asarray(extent),
            np.asarray(basis_vectors),
            self.inv_delta_radius_m,
            self.delta_radius_m,
            self.rr,
            self.gf,
            self.prod
        )


        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"Green's dyadic constructed in  {seconds_elapsed}s")

        return out

