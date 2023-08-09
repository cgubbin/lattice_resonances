import os
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slrs.types.calculations import Calculation

from slrs.types.lattice import Lattice
import numpy as np
import numpy.typing as npt
import toml
from aenum import Enum

from slrs.utils.logging import logger

class Polarisation(Enum):
    TM = 0
    TE = 1
    RightCircular = 2
    LeftCircular = 3

class Probe(Enum):
    Corner = 0
    Edge = 1
    Centre = 2
    EnsembleAverage = 3

@dataclass
class Field:
    polarisation: Polarisation
    probe: Probe

    @classmethod
    def from_file(cls, path_to_configuration_file: str) -> None:

        extension = os.path.splitext(path_to_configuration_file)[-1].lower()
        if extension != ".toml":
            raise ValueError(f"expected a `.toml` file, received a `{extension}` file")

        path_to_configuration_file = Path(".").joinpath(path_to_configuration_file)
        if not os.path.exists(path_to_configuration_file):
            raise ValueError(f"file {path_to_configuration_file} not found")

        parsed_configuration = toml.load(path_to_configuration_file)['Field']

        return cls(
            Polarisation[parsed_configuration['polarisation']],
            Probe[parsed_configuration['probe']],
        )

    @staticmethod
    def _te_field(
        slice: npt.NDArray[np.complex128],
        phi: npt.NDArray[np.complex128],
        exponent: npt.NDArray[np.complex128]
    ):
        assert slice[..., 0].shape == phi.shape, "slice and angle size must match"
        assert slice[..., 0].shape == exponent.shape, "slice and exponent size must match"
        assert slice.shape[-1] == 3, "slice must have 3 Cartesian components"

        slice[..., 0] = - np.sin(phi) * np.exp(exponent)
        slice[..., 1] = np.cos(phi) * np.exp(exponent)

    @staticmethod
    def _tm_field(
        slice: npt.NDArray[np.complex128],
        theta: npt.NDArray[np.complex128],
        phi: npt.NDArray[np.complex128],
        exponent: npt.NDArray[np.complex128]
    ):
        assert slice[..., 0].shape == theta.shape, "slice and angle size must match"
        assert slice[..., 0].shape == phi.shape, "slice and angle size must match"
        assert slice[..., 0].shape == exponent.shape, "slice and exponent size must match"
        assert slice.shape[-1] == 3, "slice must have 3 Cartesian components"

        slice[..., 0] = np.cos(phi) * np.cos(theta) * np.exp(exponent)
        slice[..., 1] = np.sin(phi) * np.cos(theta) * np.exp(exponent)
        slice[..., 2] = - np.sin(theta) * np.exp(exponent)

    @staticmethod
    def _right_circular_field(
        slice: npt.NDArray[np.complex128],
        theta: npt.NDArray[np.complex128],
        phi: npt.NDArray[np.complex128],
        exponent: npt.NDArray[np.complex128]
    ):
        assert slice[..., 0].shape == theta.shape, "slice and angle size must match"
        assert slice[..., 0].shape == phi.shape, "slice and angle size must match"
        assert slice[..., 0].shape == exponent.shape, "slice and exponent size must match"
        assert slice.shape[-1] == 3, "slice must have 3 Cartesian components"

        slice[..., 0] = 1 / np.sqrt(2) * (np.cos(phi) * np.cos(theta) + 1j * np.sin(phi)) * np.exp(exponent)
        slice[..., 1] = 1 / np.sqrt(2) * (np.sin(phi) * np.cos(theta) - 1j * np.cos(phi)) * np.exp(exponent)
        slice[..., 2] = - np.sin(theta) * np.exp(exponent) / np.sqrt(2)

    @staticmethod
    def _left_circular_field(
        slice: npt.NDArray[np.complex128],
        theta: npt.NDArray[np.complex128],
        phi: npt.NDArray[np.complex128],
        exponent: npt.NDArray[np.complex128]
    ):
        assert slice[..., 0].shape == theta.shape, "slice and angle size must match"
        assert slice[..., 0].shape == phi.shape, "slice and angle size must match"
        assert slice[..., 0].shape == exponent.shape, "slice and exponent size must match"
        assert slice.shape[-1] == 3, "slice must have 3 Cartesian components"

        slice[..., 0] = 1 / np.sqrt(2) * (np.cos(phi) * np.cos(theta) - 1j * np.sin(phi)) * np.exp(exponent)
        slice[..., 1] = 1 / np.sqrt(2) * (np.sin(phi) * np.cos(theta) + 1j * np.cos(phi)) * np.exp(exponent)
        slice[..., 2] = - np.sin(theta) * np.exp(exponent) / np.sqrt(2)


    @staticmethod
    def _fill_field(
        slice: npt.NDArray[np.complex128],
        polarisation: Polarisation,
        theta: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        exponent: npt.NDArray[np.float64],
    ):
        match polarisation:
            case Polarisation.TE:
                Field._te_field(
                    slice,
                    phi,
                    exponent
                )
            case Polarisation.TM:
                Field._tm_field(
                    slice,
                    theta,
                    phi,
                    exponent
                )
            case Polarisation.RightCircular:
                Field._right_circular_field(
                    slice,
                    theta,
                    phi,
                    exponent
                )
            case Polarisation.LeftCircular:
                Field._left_circular_field(
                    slice,
                    theta,
                    phi,
                    exponent
                )


    def _finite(
        self,
        lattice: Lattice,
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        logger.info("calculating source vector")
        start = dt.now()

        # angle of in-plane propagation in spherical coordinates
        wavevectors_im = 2. * np.pi / (wavelengths_nm * 1e-9) * lattice.background_index
        out_of_plane_wavevectors_im = np.emath.sqrt(
            wavevectors_im ** 2
                - np.linalg.norm(in_plane_wavevectors_im, axis=-1) ** 2
        )
        # angle of in-plane propagation in spherical coordinates
        phi = np.arctan2(in_plane_wavevectors_im[..., 1], in_plane_wavevectors_im[..., 0])
        # azimuthal angle from surface normal
        theta = np.arctan2(
            np.sqrt(in_plane_wavevectors_im[..., 0] ** 2 + in_plane_wavevectors_im[..., 1] ** 2),
            np.where(np.imag(out_of_plane_wavevectors_im) == 0.0, np.real(out_of_plane_wavevectors_im), 0.0)
        )

        # electric field stacked in the same way as the wavevector / energy grid with the last
        # axis for the 3 cartesian components and the one-before last for the N sites
        particle_count = lattice.particle_count()
        electric_field = np.zeros((*wavelengths_nm.shape, particle_count * 3), dtype=np.complex128)

        # TODO add a method to `gen` this iterator on `Lattice`
        match self.probe:
            case Probe.EnsembleAverage:
                for ii in tqdm(range(particle_count), leave=False):
                    position_m = sum(
                        i * a for (i, a) in zip(
                            [ii % lattice.finite_extent[0], ii // lattice.finite_extent[0]],
                            lattice.basis_vectors
                        )
                    )
                    exponent = 1j * np.dot(in_plane_wavevectors_im, position_m)
                    self._fill_field(
                        electric_field[..., 3 * ii: 3 * (ii + 1)],
                        self.polarisation,
                        theta,
                        phi,
                        exponent
                    )
            case Probe.Corner:
                ii = 0
                position_m = sum(
                    i * a for (i, a) in zip(
                        [ii % lattice.finite_extent[0], ii // lattice.finite_extent[0]],
                        lattice.basis_vectors
                    )
                )
                exponent = 1j * np.dot(in_plane_wavevectors_im, position_m)
                self._fill_field(
                    electric_field[..., 3 * ii: 3 * (ii + 1)],
                    self.polarisation,
                    theta,
                    phi,
                    exponent
                )
            case Probe.Edge:
                if lattice.finite_extent[0] % 2 == 1:
                    # if odd get the center of the edge
                    iis = [lattice.finite_extent[0] // 2]
                else:
                    iis = [
                        lattice.finite_extent[0] // 2 - 1,
                        lattice.finite_extent[0] // 2
                    ]
                    # else average the two central points
                for ii in iis:
                    ii = 0
                    position_m = sum(
                        i * a for (i, a) in zip(
                            [ii % lattice.finite_extent[0], ii // lattice.finite_extent[0]],
                            lattice.basis_vectors
                        )
                    )
                    exponent = 1j * np.dot(in_plane_wavevectors_im, position_m)
                    self._fill_field(
                        electric_field[..., 3 * ii: 3 * (ii + 1)],
                        self.polarisation,
                        theta,
                        phi,
                        exponent
                    )
            case Probe.Centre:
                if lattice.finite_extent[0] % 2 == 1:
                    # if odd get the center of the edge
                    if lattice.finite_extent[1] % 2 == 1:
                        iis = [
                            lattice.finite_extent[0] // 2 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0]
                        ]
                    else:
                        iis = [
                            lattice.finite_extent[0] // 2 + (lattice.finite_extent[1] // 2 - 1) * lattice.finite_extent[0],
                            lattice.finite_extent[0] // 2 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0],
                        ]
                else:
                    if lattice.finite_extent[1] % 2 == 1:
                        iis = [
                            lattice.finite_extent[0] // 2 - 1 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0],
                            lattice.finite_extent[0] // 2 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0]
                        ]
                    else:
                        iis = [
                            lattice.finite_extent[0] // 2 + (lattice.finite_extent[1] // 2 - 1) * lattice.finite_extent[0],
                            lattice.finite_extent[0] // 2 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0],
                            lattice.finite_extent[0] // 2 - 1 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0],
                            lattice.finite_extent[0] // 2 + lattice.finite_extent[1] // 2 * lattice.finite_extent[0]
                        ]

                for ii in iis:
                    ii = 0
                    position_m = sum(
                        i * a for (i, a) in zip(
                            [ii % lattice.finite_extent[0], ii // lattice.finite_extent[0]],
                            lattice.basis_vectors
                        )
                    )
                    exponent = 1j * np.dot(in_plane_wavevectors_im, position_m)
                    self._fill_field(
                        electric_field[..., 3 * ii: 3 * (ii + 1)],
                        self.polarisation,
                        theta,
                        phi,
                        exponent
                    )

        electric_field = np.where(theta[..., np.newaxis] != np.pi / 2, electric_field, 0.0)


        seconds_elapsed = (dt.now() - start).total_seconds()
        logger.success(f"calculated source vector in {seconds_elapsed}")
        return electric_field

    def _infinite(
        self,
        lattice: Lattice,
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:

        # todo -> add in sphp
        wavevectors_im = 2. * np.pi / (wavelengths_nm * 1e-9) * lattice.coupled_mode_index(wavelengths_nm)
        out_of_plane_wavevectors_im = np.emath.sqrt(
            wavevectors_im ** 2
                - np.linalg.norm(in_plane_wavevectors_im, axis=-1) ** 2
        )
        # angle of in-plane propagation in spherical coordinates
        phi = np.arctan2(in_plane_wavevectors_im[..., 1], in_plane_wavevectors_im[..., 0])
        # azimuthal angle from surface normal
        theta = np.arctan2(
            np.sqrt(in_plane_wavevectors_im[..., 0] ** 2 + in_plane_wavevectors_im[..., 1] ** 2),
            np.where(np.imag(out_of_plane_wavevectors_im) == 0.0, np.real(out_of_plane_wavevectors_im), 0)
        )

        stacked_field = np.zeros((*wavelengths_nm.shape, 6), dtype=np.complex128)

        match self.polarisation:
            case Polarisation.TE:
                stacked_field[..., 0] = - np.sin(phi)
                stacked_field[..., 1] = np.cos(phi)
                stacked_field[..., 3] = np.cos(phi) * np.cos(theta)
                stacked_field[..., 4] = np.sin(phi) * np.cos(theta)
                stacked_field[..., 5] = - np.sin(theta)
            case Polarisation.TM:
                stacked_field[..., 0] = np.cos(phi) * np.cos(theta)
                stacked_field[..., 1] = np.sin(phi) * np.cos(theta)
                stacked_field[..., 2] = - np.sin(theta)
                stacked_field[..., 3] = - np.sin(phi)
                stacked_field[..., 4] = np.cos(phi)
            case Polarisation.RightCircular:
                stacked_field[..., 0] = 1 / np.sqrt(2) * (np.cos(phi) * np.cos(theta) + 1j * np.sin(phi))
                stacked_field[..., 1] = 1 / np.sqrt(2) * (np.sin(phi) * np.cos(theta) - 1j * np.cos(phi))
                stacked_field[..., 2] = - np.sin(theta) / np.sqrt(2)
            case Polarisation.LeftCircular:
                stacked_field[..., 0] = 1 / np.sqrt(2) * (np.cos(phi) * np.cos(theta) - 1j * np.sin(phi))
                stacked_field[..., 1] = 1 / np.sqrt(2) * (np.sin(phi) * np.cos(theta) + 1j * np.cos(phi))
                stacked_field[..., 2] = - np.sin(theta) / np.sqrt(2)


        stacked_field = np.where(theta[..., np.newaxis] != np.pi / 2, stacked_field, 0.0)

        if (positions_in_cell_nm := lattice.positions_in_cell()) is not None:
            for positions_nm in positions_in_cell_nm:
                stacked_field = np.concatenate([
                    stacked_field,
                    stacked_field * np.exp(1j * np.dot(in_plane_wavevectors_im, 1e-9 * positions_nm))[..., np.newaxis]
                    ],
                    axis=-1
                )
        return stacked_field

    def generate(
        self,
        lattice: Lattice,
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        if lattice.finite_extent is not None:
            return self._finite(lattice, in_plane_wavevectors_im, wavelengths_nm)
        else:
            return self._infinite(lattice, in_plane_wavevectors_im, wavelengths_nm)

    def illuminated_count(
        self,
        lattice: Lattice,
    ) -> int:
        match self.probe:
            case Probe.EnsembleAverage:
                if (particle_count := lattice.particle_count()) is not None:
                    return particle_count
                else:
                    raise ValueError("infinite particles are illuminated, we cannot count them")
            case Probe.Corner:
                return 1
            case Probe.Edge:
                return 1
            case Probe.Centre:
                return 1




