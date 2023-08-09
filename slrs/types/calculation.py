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
from scipy.constants import e, epsilon_0, hbar, mu_0, speed_of_light
from slrs.dyadics.direct import DirectDyadic
from slrs.dyadics.ewald import EwaldDyadic
from slrs.types.field import Field
from slrs.types.lattice import Lattice
from slrs.types.lattice import Symmetry
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
        """
        Returns the wavelength grid in nanometres.

        To get a uniform grid in the preferred unit the grid is generated as a
        linspace in that unit, then cast back to nanometres.
        """
        match self.preferred_unit:
            case SpectralUnit.Nanometres:
                return np.linspace(self.minimum_wavelength_nm, self.maximum_wavelength_nm, self.number_of_wavelengths)
            case SpectralUnit.Wavenumbers:
                return wavenumber_to_nanometre(
                    np.linspace(
                        nanometre_to_wavenumber(self.maximum_wavelength_nm),
                        nanometre_to_wavenumber(self.minimum_wavelength_nm),
                        self.number_of_wavelengths
                    )
                )
            case SpectralUnit.ElectronVolts:
                return ev_to_nanometre(
                    np.linspace(
                        nanometre_to_ev(self.maximum_wavelength_nm),
                        nanometre_to_ev(self.minimum_wavelength_nm),
                        self.number_of_wavelengths
                    )
                )

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
            ] = single_particle_inverse_polarisability_tensor[:, 0, :3, :3]

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

        if self.lattice.symmetry == Symmetry.Honeycomb:
            self.single_particle_inverse_polarisability = np.block([
                [single_particle_inverse_polarisability, np.zeros_like(single_particle_inverse_polarisability)],
                [np.zeros_like(single_particle_inverse_polarisability), single_particle_inverse_polarisability],
            ])
        else:
            self.single_particle_inverse_polarisability = single_particle_inverse_polarisability


        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"inverse polarisability tensor built in {seconds_elapsed:.3f}s")

        start = dt.now()
        logger.info("constructing field source vector")
        self.source_vector = self.field.generate(
            self.lattice,
            in_plane_wavevectors_im,
            wavelengths_nm
        )
        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"source vector constructed in {seconds_elapsed:.3f}s")


        builder = EwaldDyadic()
        cutoff = 5
        evaluation_point = np.array([0.0, 0.0, 0.0])

        start = dt.now()
        logger.info("constructing Green's Dyadic")
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
        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"dyadic constructed in {seconds_elapsed:.3f}s")

    @staticmethod
    @jit(nopython=True, parallel=True)
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

        if (positions_in_cell_nm := self.lattice.positions_in_cell()) is None:
            self.solution_vector = np.zeros((*wavevectors_im.shape, 6), dtype=np.complex128)
        else:
            self.solution_vector = np.zeros((*wavevectors_im.shape, 6 * (positions_in_cell_nm.shape[0] + 1)), dtype=np.complex128)

        start = dt.now()

        for ll in tqdm(range(wavevectors_im.shape[1]), leave=False):
            self.solution_vector[:, ll] = np.linalg.solve(
                (
                    self.single_particle_inverse_polarisability
                        - self.matrix_a[:, ll, ...]
                ),
                self.source_vector[:, ll, ...]
            )

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
        diffracted_order_in_plane_wavevectors_inm: npt.NDArray[np.complex128],
        out_of_plane_wavevectors_inm: npt.NDArray[np.complex128],
        wavevectors_inm: npt.NDArray[np.complex128],
        reflectance: bool
    ) -> npt.NDArray[np.complex128]:
        matrix_d = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)
        matrix_o = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)


        theta = np.arctan2(
            np.linalg.norm(diffracted_order_in_plane_wavevectors_inm, axis=-1),
            np.where(np.imag(out_of_plane_wavevectors_inm) == 0.0, np.real(out_of_plane_wavevectors_inm), 0)
        )
        if reflectance:
            theta = np.pi - theta
        phi = np.arctan2(
            diffracted_order_in_plane_wavevectors_inm[..., 1],
            diffracted_order_in_plane_wavevectors_inm[..., 0]
        )

        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)

        matrix_d[..., 0, 0] = 1 - x ** 2
        matrix_d[..., 0, 1] = - x * y
        matrix_d[..., 0, 2] = - x * z
        matrix_d[..., 1, 0] = - x * y
        matrix_d[..., 1, 1] = 1 - y ** 2
        matrix_d[..., 1, 2] = - y * z
        matrix_d[..., 2, 2] = - x * z
        matrix_d[..., 2, 2] = - y * z
        matrix_d[..., 2, 2] = 1 - z ** 2

        matrix_o[..., 0, 1] = - z
        matrix_o[..., 0, 2] = y
        matrix_o[..., 1, 0] = z
        matrix_o[..., 1, 2] = - x
        matrix_o[..., 2, 0] = - y
        matrix_o[..., 2, 1] = x


        return np.block([
            [matrix_d, matrix_o],
            [-matrix_o, matrix_d],
        ])


    def _extinction_infinite(self, reflectance: bool = True) -> npt.NDArray[np.complex128]:
        # We actually find the reflectance...
        # the polarisations in the unit cell are stored in the solution_vector

        # find the vector F_m_n
        cutoff = 0

        in_plane_wavevectors_im, wavelengths_nm = self.grid.generate()

        in_plane_wavevectors_inm = in_plane_wavevectors_im * 1e-9
        far_field_wavevectors_inm = 2 * np.pi / wavelengths_nm

        # The location of the measurement in nanometres
        evaluation_point_nm = np.zeros(3)

        # electric field, followed by magnetic field
        scattered_far_field = np.zeros((*wavelengths_nm.shape, 6), dtype=np.complex128)

        wavevectors_inm = 2 * np.pi / wavelengths_nm * self.lattice.coupled_mode_index(wavelengths_nm)

        unit_cell_area_nm2 = self.lattice.unit_cell_area() / (1e-18)


        for m, n in product(range(-cutoff, cutoff + 1), range(-cutoff, cutoff + 1)):
            reciprocal_wavevectors_inm = sum(
                i * b * 1e-9 for i, b in zip([m, n], self.lattice.reciprocal_vectors)
            )
            diffracted_order_in_plane_wavevectors_inm = in_plane_wavevectors_inm + reciprocal_wavevectors_inm


            out_of_plane_wavevectors_inm = np.emath.sqrt(
                wavevectors_inm ** 2
                    - np.linalg.norm(
                        diffracted_order_in_plane_wavevectors_inm,
                        axis=-1
                    ) ** 2
            )

            x = diffracted_order_in_plane_wavevectors_inm[..., 0] / wavevectors_inm
            y = diffracted_order_in_plane_wavevectors_inm[..., 1] / wavevectors_inm
            z = out_of_plane_wavevectors_inm / wavevectors_inm

            theta = np.arctan2(
                np.linalg.norm(diffracted_order_in_plane_wavevectors_inm, axis=-1),
                np.where(np.imag(out_of_plane_wavevectors_inm) == 0.0, np.real(out_of_plane_wavevectors_inm), 0)
            )
            if reflectance:
                theta = np.pi - theta

            phi = np.arctan2(
                diffracted_order_in_plane_wavevectors_inm[..., 1],
                diffracted_order_in_plane_wavevectors_inm[..., 0]
            )
            x = np.cos(phi) * np.sin(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(theta)

            m = self._matrix_m(
                diffracted_order_in_plane_wavevectors_inm,
                out_of_plane_wavevectors_inm,
                wavevectors_inm,
                reflectance
            )

            # impl Eq 35
            summed_polarisation = self.solution_vector[..., 0:6] * np.exp(
                                1j * np.dot(diffracted_order_in_plane_wavevectors_inm, evaluation_point_nm)
            )[..., np.newaxis]
            if (positions_in_cell_nm := self.lattice.positions_in_cell()) is not None:
                for ii, positions_nm in enumerate(positions_in_cell_nm):
                    summed_polarisation += self.solution_vector[..., 6*(ii + 1):6*(ii + 2)] * np.exp(
                                1j * np.dot(diffracted_order_in_plane_wavevectors_inm, evaluation_point_nm - positions_nm)
                            )[..., np.newaxis]
            result = summed_polarisation * (
                    wavevectors_inm[..., np.newaxis] ** 2
                        * 2 * np.pi * 1j
                        / unit_cell_area_nm2
                        / out_of_plane_wavevectors_inm[..., np.newaxis]
            ) / 1e-27


            result = np.einsum(
                "...ij,...j",
                m, result
            )

            scattered_far_field += np.where(
                np.imag(out_of_plane_wavevectors_inm)[..., np.newaxis] == 0.0, result, 0.0
            )

        # Add zero-order input field contribution
        self.scattered_far_field = scattered_far_field * 0.0 + self.source_vector[..., :6]
        self.total_source = self.source_vector[..., :6]

        if (positions_in_cell_nm := self.lattice.positions_in_cell()) is not None:
            for ii in range(positions_in_cell_nm.shape[0]):
                self.scattered_far_field += self.source_vector[...,6*(ii+1):6*(ii+2)]
                self.total_source += self.source_vector[...,6*(ii+1):6*(ii+2)]

        denominator = - np.real(
            self.total_source[..., 0] * self.total_source[..., 4]
                - self.total_source[..., 1] * self.total_source[..., 3]
        )

        renormalised_denominator = np.where(
            denominator != 0.0,
            denominator,
            1e10
        )


        self.scattered_far_field = scattered_far_field




        z = np.sqrt(epsilon_0 * self.lattice.background_index ** 2 / mu_0)

        if reflectance:
            factor = -1
        else:
            factor = 1

        return factor * np.real(
            self.scattered_far_field[..., 0] * self.scattered_far_field[..., 4]
                - self.scattered_far_field[..., 1] * self.scattered_far_field[..., 3]
        ) / renormalised_denominator





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



