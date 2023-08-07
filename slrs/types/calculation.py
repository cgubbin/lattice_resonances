from dataclasses import dataclass
import colorlog
import logging
import os
from datetime import datetime as dt
from itertools import product
from pathlib import Path
from tqdm import tqdm
from typing import List
from typing import Tuple

from time import sleep

import numpy as np
import numpy.typing as npt
import toml
from aenum import Enum
from numba import jit, prange, set_num_threads
from numba_progress import ProgressBar
from scipy.constants import e, hbar, speed_of_light
from slrs.dyadics.direct import DirectDyadic
from slrs.dyadics.ewald import EwaldDyadic
from slrs.types.field import Field
from slrs.types.lattice import Lattice
from slrs.types.particle import Particle
from slrs.utils.conversions import ev_to_nanometre, nanometre_to_ev, nanometre_to_wavenumber, wavenumber_to_nanometre
from slrs.utils.logging import logger


class SpectralUnit:
    Nanometres = 0
    Wavenumbers = 1
    ElectronVolts = 2



@dataclass
class Grid:
    minimum_wavevector_im: float
    maximum_wavevector_im: float
    minimum_wavelength_nm: float
    maximum_wavelength_nm: float
    number_of_wavevectors: float
    number_of_wavelengths: float
    preferred_unit: SpectralUnit

    @classmethod
    def from_file(cls, path_to_configuration_file: str) -> None:
        extension = os.path.splitext(path_to_configuration_file)[-1].lower()
        if extension != ".toml":
            raise ValueError(f"expected a `.toml` file, received a `{extension}` file")

        path_to_configuration_file = Path(".").joinpath(path_to_configuration_file)
        if not os.path.exists(path_to_configuration_file):
            raise ValueError(f"file {path_to_configuration_file} not found")

        parsed_configuration = toml.load(path_to_configuration_file)['Grid']

        if (minimum_wavelength_nm := parsed_configuration.get('minimum_wavelength_nm')) is not None \
            and (maximum_wavelength_nm := parsed_configuration.get('maximum_wavelength_nm')) is not None:
            return cls(
                parsed_configuration['minimum_wavevector_im'],
                parsed_configuration['maximum_wavevector_im'],
                minimum_wavelength_nm,
                maximum_wavelength_nm,
                parsed_configuration['number_of_wavevectors'],
                parsed_configuration['number_of_wavelengths'],
                SpectralUnit.Nanometres
            )

        if (minimum_wavenumber_icm := parsed_configuration.get('minimum_wavenumber_icm')) is not None \
            and (maximum_wavenumber_icm := parsed_configuration.get('maximum_wavenumber_icm')) is not None:
            return cls(
                parsed_configuration['minimum_wavevector_im'],
                parsed_configuration['maximum_wavevector_im'],
                wavenumber_to_nanometre(minimum_wavenumber_icm),
                wavenumber_to_nanometre(maximum_wavenumber_icm),
                parsed_configuration['number_of_wavevectors'],
                parsed_configuration['number_of_wavelengths'],
                SpectralUnit.Wavenumbers
            )

        if (minimum_energy_ev := parsed_configuration.get('minimum_energy_ev')) is not None \
            and (maximum_energy_ev := parsed_configuration.get('maximum_energy_ev')) is not None:
            return cls(
                parsed_configuration['minimum_wavevector_im'],
                parsed_configuration['maximum_wavevector_im'],
                ev_to_nanometre(minimum_energy_ev),
                ev_to_nanometre(maximum_energy_ev),
                parsed_configuration['number_of_wavevectors'],
                parsed_configuration['number_of_wavelengths'],
                SpectralUnit.ElectronVolts
            )

    def wavevectors_im(self) -> npt.NDArray[np.float64]:
        return np.linspace(self.minimum_wavevector_im, self.maximum_wavevector_im, self.number_of_wavevectors)

    def wavelengths_nm(self) -> npt.NDArray[np.float64]:
        return np.linspace(self.minimum_wavelength_nm, self.maximum_wavelength_nm, self.number_of_wavelengths)

    def generate(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        wavevectors_x, wavelengths_nm = np.meshgrid(
            self.wavevectors_im(), self.wavelengths_nm()
        )
        wavevectors_y = np.zeros_like(wavevectors_x)
        wavevectors_z = np.zeros_like(wavevectors_x)

        full_wavevector = np.concatenate((
            wavevectors_x[..., np.newaxis],
            wavevectors_y[..., np.newaxis],
            wavevectors_z[..., np.newaxis]
            ),
            axis=2
        )

        return full_wavevector, wavelengths_nm

    def plot_grid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        in_plane_wavevectors_im, wavelengths_nm = self.generate()

        match self.preferred_unit:
            case SpectralUnit.Nanometres:
                return in_plane_wavevectors_im[..., 0], wavelengths_nm
            case SpectralUnit.Wavenumbers:
                return in_plane_wavevectors_im[..., 0], nanometre_to_wavenumber(wavelengths_nm)
            case SpectralUnit.ElectronVolts:
                return in_plane_wavevectors_im[..., 0], nanometre_to_ev(wavelengths_nm)




@dataclass
class Calculation:
    name: str
    grid: Grid
    lattice: Lattice
    particle: Particle
    field: Field

    @classmethod
    def from_file(cls, path_to_configuration_file: str) -> None:
        return cls(
            os.path.basename(Path(path_to_configuration_file)),
            Grid.from_file(path_to_configuration_file),
            Lattice.from_file(path_to_configuration_file),
            Particle.from_file(path_to_configuration_file),
            Field.from_file(path_to_configuration_file)
        )

    def plot_grid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.grid.plot_grid()

    def _build_finite(self, extent: List[int]):
        # Build the Green's dyadic
        in_plane_wavevectors_im, wavelengths_nm = self.grid.generate()
        num_particles = extent[0] * extent[1]
        self.lattice_inverse_polarisability = np.zeros(
            (wavelengths_nm.shape[0], num_particles * 3, num_particles * 3),
            dtype=np.complex128
        )


        start = dt.now()
        logger.info("constructing single particle polarisability tensor")
        single_particle_inverse_polarisability_tensor = self.particle.inverse_polarisability_tensor(
            wavelengths_nm,
            self.lattice.background_index
        )

        logger.info("constructing lattice polarisability tensor")

        for ii in range(extent[0] * extent[1]):
            self.lattice_inverse_polarisability[
                ..., 3*ii:3*(ii+1), 3*ii:3*(ii+1)
            ] = single_particle_inverse_polarisability_tensor[:, 0, ...]

        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"inverse polarisability tensor built in {seconds_elapsed:.3f}s")


        self.source_vector = self.field.generate(
            self.lattice,
            in_plane_wavevectors_im,
            wavelengths_nm
        )

        # set up Eq. 4
        builder = DirectDyadic()
        self.matrix_a = builder.construct(
            extent,
            self.lattice.basis_vectors,
            self.lattice.coupled_mode_index(wavelengths_nm),
            wavelengths_nm
        )


    def _build_infinite(self):
        # Build the Green's dyadic
        in_plane_wavevectors_im, wavelengths_nm = self.grid.generate()


        start = dt.now()
        logger.info("constructing single particle polarisability tensor")
        single_particle_inverse_polarisability = self.particle.inverse_polarisability_tensor(
            wavelengths_nm,
            self.lattice.background_index
        )[:, 0, ...]

        self.single_particle_inverse_polarisability = np.block([
            [single_particle_inverse_polarisability, np.zeros_like(single_particle_inverse_polarisability)],
            [np.zeros_like(single_particle_inverse_polarisability), single_particle_inverse_polarisability],
        ])


        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"inverse polarisability tensor built in {seconds_elapsed:.3f}s")

        self.source_vector = self.field.generate(
            self.lattice,
            in_plane_wavevectors_im,
            wavelengths_nm
        )


        builder = EwaldDyadic()
        cutoff = 5
        evaluation_point = np.array([0.0, 0.0, 0.0])

        self.matrix_a = builder.construct(
            self.lattice.basis_vectors,
            self.lattice.reciprocal_vectors,
            self.lattice.unit_cell_area(),
            self.lattice.positions_in_cell(),
            self.lattice.coupled_mode_index(wavelengths_nm),
            in_plane_wavevectors_im,
            wavelengths_nm,
            evaluation_point,
            cutoff,
            include_origin=False
        )

    @staticmethod
    @jit(nopython=True)
    def _solve_finite_inner(
        solution_vector,
        lattice_inverse_polarisability,
        matrix_a,
        source_vector,
        unique_wavevectors_im,
        progress_proxy
    ):
        #start = dt.now()
        for ll in prange(solution_vector.shape[1]):
            for kk in range(solution_vector.shape[0]):
                solution_vector[kk, ll] = np.linalg.solve(
                    (
                        lattice_inverse_polarisability[kk]
                            - matrix_a[kk] * unique_wavevectors_im[kk] ** 2 # / coupled_mode_index[] ** 2
                    ),
                    source_vector[kk, ll, ...]
                )
            progress_proxy.update(1)

        #seconds_elapsed = (dt.now() - start).total_seconds()
        #logger.success(f"linear system solved in {seconds_elapsed:.3f}s")



    def _solve_finite(self, extent: List[int]):
        _, wavelengths_nm = self.grid.generate()
        num_sites = extent[0] * extent[1]

        coupled_mode_index = self.lattice.coupled_mode_index(wavelengths_nm)
        wavevectors_im = 2 * np.pi / (1e-9 * wavelengths_nm) # * coupled_mode_index
        unique_wavevectors_im = wavevectors_im[:, 0]

        self.solution_vector = np.zeros((*wavevectors_im.shape, 3 * num_sites), dtype=np.complex128)

        start = dt.now()

        ## Preferred solution with working monitor, but single threading

        # for ll in tqdm(range(wavevectors_im.shape[1]), leave=False):
        #     self.solution_vector[:, ll] = np.linalg.solve(
        #         (
        #             self.lattice_inverse_polarisability
        #                 - self.matrix_a * unique_wavevectors_im[..., np.newaxis, np.newaxis] ** 2 # / coupled_mode_index[] ** 2
        #         ),
        #         self.source_vector[:, ll, ...]
        #     )

        set_num_threads(8)
        with ProgressBar(total=wavevectors_im.shape[1], dynamic_ncols=True, leave=False) as progress:
            self._solve_finite_inner(
                self.solution_vector,
                self.lattice_inverse_polarisability,
                self.matrix_a,
                self.source_vector,
                unique_wavevectors_im,
                progress
            )

        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"linear system solved in {seconds_elapsed:.3f}s")

    def _solve_infinite(self):
        _, wavelengths_nm = self.grid.generate()

        coupled_mode_index = self.lattice.coupled_mode_index(wavelengths_nm)
        wavevectors_im = 2 * np.pi / (1e-9 * wavelengths_nm) # * coupled_mode_index
        unique_wavevectors_im = wavevectors_im[:, 0]

        self.solution_vector = np.zeros((*wavevectors_im.shape, 6), dtype=np.complex128)
        self.test= np.zeros_like(wavevectors_im, dtype=np.float64)

        start = dt.now()

        for ll in tqdm(range(wavevectors_im.shape[1]), leave=False):
            # self.solution_vector[:, ll] = np.linalg.solve(
            #     (
            #         self.single_particle_inverse_polarisability
            #             - self.matrix_a[:, ll, ...]
            #     ),
            #     self.source_vector[:, ll, ...]
            # )
            inv_mat = (#np.linalg.inv(
                    # self.single_particle_inverse_polarisability
                        - self.matrix_a[:, ll, ...]
                )
            self.test[..., ll] = np.real(self.matrix_a[:, ll, 1, 1]) #+ inv_mat[..., 1, 1] + inv_mat[..., 2, 2]

        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"linear system solved in {seconds_elapsed:.3f}s")


    def _build(self):
        """
        Build the SLP matrix
        """
        # is the calculation finite or infinit
        if (extent := self.lattice.finite_extent) is not None:
            logger.info(f"building problem for a {extent[0]}x{extent[1]} lattice")
            self._build_finite(extent)
            logger.info(f"solving problem for a {extent[0]}x{extent[1]} lattice")
            self._solve_finite(extent)
        else:
            logger.info(f"building problem for an infinite lattice")
            self._build_infinite()
            logger.info(f"solving problem for an infinite lattice")
            self._solve_infinite()

    def _extinction_finite(self, extent: List[int]) -> npt.NDArray[np.complex128]:

        _, wavelengths_nm = self.grid.generate()
        wavevectors_im = 2. * np.pi * self.lattice.background_index / (1e-9 * wavelengths_nm)

        # sum over all lattice points...
        extinction = 4 * np.pi * wavevectors_im * np.imag(np.sum(
            np.conj(self.source_vector) * self.solution_vector,
            axis=-1
        )) / self.field.illuminated_count(self.lattice)

        return extinction

    @staticmethod
    def _matrix_m(
        x: npt.NDArray[np.complex128],
        y: npt.NDArray[np.complex128],
        z: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        matrix_m = np.zeros((*x.shape, 3, 3), dtype=np.complex128)
        matrix_o = np.zeros((*x.shape, 3, 3), dtype=np.complex128)

        matrix_m[..., 0, 0] = 1 - x ** 2
        matrix_m[..., 0, 1] = - x * y
        matrix_m[..., 0, 2] = - x * z
        matrix_m[..., 1, 0] = - x * y
        matrix_m[..., 1, 1] = 1 - y ** 2
        matrix_m[..., 1, 2] = - y * z
        matrix_m[..., 2, 2] = - x * z
        matrix_m[..., 2, 2] = - y * z
        matrix_m[..., 2, 2] = 1 - z ** 2

        matrix_o[..., 0, 1] = - z
        matrix_o[..., 0, 2] = y
        matrix_o[..., 1, 0] = z
        matrix_o[..., 1, 2] = - x
        matrix_o[..., 2, 0] = - y
        matrix_o[..., 2, 1] = x


        return np.block([
            [matrix_m, matrix_o],
            [matrix_o, matrix_m],
        ])


    def _extinction_infinite(self) -> npt.NDArray[np.complex128]:
        # We actually find the reflectance...
        # the polarisations in the unit cell are stored in the solution_vector

        # find the vector F_m_n
        # cutoff = 5

        # in_plane_wavevectors_im, wavelengths_nm = self.grid.generate()

        # ff_wavevectors_im = 2 * np.pi / (1e-9 * wavelengths_nm)
        # rl_ms = [np.zeros(3)]

        # stacked_field_vector = np.zeros((*wavelengths_nm.shape, 6), dtype=np.complex128)

        # wavevectors_im = 2 * np.pi / (wavelengths_nm * 1e-9)

        # for m, n in product(range(-cutoff, cutoff + 1), range(-cutoff, cutoff + 1)):
        #     bragg_wavevector_im = sum(
        #         i * b for i, b in zip([m, n], self.lattice.reciprocal_vectors)
        #     )
        #     diffracted_order_in_plane_wavevectors_im = in_plane_wavevectors_im
        #     diffracted_order_in_plane_wavevectors_im[..., 0] + bragg_wavevector_im[0]
        #     diffracted_order_in_plane_wavevectors_im[..., 1] + bragg_wavevector_im[1]

        #     kz = np.emath.sqrt(
        #         wavevectors_im ** 2
        #             - diffracted_order_in_plane_wavevectors_im[..., 0] ** 2
        #             - diffracted_order_in_plane_wavevectors_im[..., 1] ** 2
        #     )

        #     # exponent = - 1j * np.dot(diffracted_order_in_plane_wavevectors_im, rl_m)

        #     x = diffracted_order_in_plane_wavevectors_im[..., 0] / wavevectors_im
        #     y = diffracted_order_in_plane_wavevectors_im[..., 1] / wavevectors_im
        #     z = kz / wavevectors_im
        #     m = self._matrix_m(x, y, z)
        #     result = sum(
        #             self.solution_vector * np.exp(- 1j * np.dot(diffracted_order_in_plane_wavevectors_im, rl_m))[..., np.newaxis]
        #             for rl_m in rl_ms
        #         ) * wavevectors_im[..., np.newaxis] * 2 * np.pi * 1j / self.lattice.unit_cell_area() / kz[..., np.newaxis]


        #     stacked_field_vector += np.einsum(
        #         "...ij,...j",
        #         m, result
        #     )

        # self.stacked_field_vector = stacked_field_vector


        return (self.test)
        # return np.real(
        #     self.stacked_field_vector[..., 0] * self.stacked_field_vector[..., 4]
        #         - self.stacked_field_vector[..., 1] * self.stacked_field_vector[..., 3]
        # )





    def _extinction(self) -> npt.NDArray[np.complex128]:
        """
        Returns the extinction cross section for the lattice.
        """
        if (extent := self.lattice.finite_extent) is not None:
            extinction = self._extinction_finite(extent) / (np.pi * (1e-9 * self.particle.radius_nm) ** 2)
        else:
            extinction = self._extinction_infinite()

        return extinction



    def cross_sections(self) -> npt.NDArray[np.complex128]:
        try:
            logger.info(f"beginning build step for {self.name}")
            self._build()
        except:
            logger.info("Unexpected exception in build step", exc_info=True)
            raise

        try:
            logger.info(f"beginning solve step for {self.name}")
            extinction = self._extinction()
        except:
            logger.info("Unexpected exception in solve step", exc_info=True)
            raise


        # absorption = np.zeros_like(wavelengths_nm, dtype=np.complex128)
        # single_particle_inverse_polarisability_tensor = self.particle.inverse_polarisability_tensor(self)
        # for ii in range(self.lattice.particle_count()):
        #     polarisation = self.solution_vector[..., 3 * ii : 3 * (ii + 1)]
        #     absorption += (
        #        np.dot(polarisation.T, np.dot(np.conj(single_particle_inverse_polarisability_tensor), polarisation))
        #        - 2 / 3 * wavevectors_im ** 3 * np.dot(np.conj(polarisation.T), polarisation)
        #    )
        # sigma_abs = 4 * np.real(absorption) * np.pi * wavevectors_im / num_dipoles

        return extinction


    def ewald_greens_function(self) -> npt.NDArray[np.complex128]:
        self._build_infinite()
        return self.matrix_a

    def effective_polarisabilities(self) -> npt.NDArray[np.complex128]:
        self._build_infinite()
        effective_polarisabilities = np.zeros_like(self.matrix_a, dtype=np.complex128)

        for ll in range(self.matrix_a.shape[1]):
            effective_polarisabilities[:, ll, ...] = np.linalg.inv(
                self.single_particle_inverse_polarisability
                    - self.matrix_a[:, ll, ...]
            )

        broadcast_single_particle = np.zeros_like(self.matrix_a, dtype=np.complex128)
        single_particle = np.linalg.inv(self.single_particle_inverse_polarisability)
        for ll in range(self.matrix_a.shape[1]):
            broadcast_single_particle[:, ll, ...] = single_particle

        return (broadcast_single_particle, effective_polarisabilities)



