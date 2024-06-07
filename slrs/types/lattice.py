import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from typing import Optional

import toml
import numpy as np
import numpy.typing as npt
from aenum import Enum

from slrs.utils.conversions import nanometre_to_wavenumber


class Symmetry(Enum):
    Rectangular = 0
    Hexagonal = 1
    Honeycomb = 2


def epsilon_sic(
    wavelength_nm: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    wavelength_nm = np.asarray(wavelength_nm)

    wavenumber_icm = nanometre_to_wavenumber(wavelength_nm)

    lo = 970.0
    to = 797.5
    gamma = 4.0
    eps_inf = 6.52

    return eps_inf * (
        (lo**2 - wavenumber_icm * (wavenumber_icm + 1j * gamma))
        / (to**2 - wavenumber_icm * (wavenumber_icm + 1j * gamma))
    )


# The effective index of the SPP on an SiC substrate
def n_spp(wavelength_nm: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    eps_sub = epsilon_sic(wavelength_nm)
    result = np.emath.sqrt(eps_sub / (eps_sub + 1))

    return result


@dataclass
class Lattice:
    symmetry: Symmetry
    lengths_nm: List[float]
    background_index: float
    finite_extent: Optional[List[int]]
    basis_vectors: List[npt.NDArray[np.float64]] = field(init=False)
    reciprocal_vectors: List[npt.NDArray[np.float64]] = field(init=False)
    # Should coupling between resonators use an SPP or, a if False a free-photon
    spp_coupling: bool

    def __post_init__(self):
        # 90 degree rotation matrix
        r_mat = np.array(
            [
                [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
            ]
        )
        r_mat = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )
        match self.symmetry:
            case Symmetry.Rectangular:
                assert (
                    len(self.lengths_nm) == 2
                ), "a rectangular lattice is characterised by two lengthr"

                self.basis_vectors = [
                    np.array([self.lengths_nm[0] * 1e-9, 0, 0]),
                    np.array([0, self.lengths_nm[1] * 1e-9, 0]),
                ]
            case Symmetry.Hexagonal:
                assert (
                    len(self.lengths_nm) == 1
                ), "a hexagonal lattice is characterised by a single length"

                a = self.lengths_nm[0] * 1e-9

                self.basis_vectors = [
                    a * np.array([1.0, 0.0, 0.0]),
                    a * np.array([0.5, np.sqrt(3.0) / 2.0, 0.0]),
                ]
            case Symmetry.Honeycomb:
                assert (
                    len(self.lengths_nm) == 1
                ), "a hexagonal lattice is characterised by a single length"

                a = self.lengths_nm[0] * 1e-9

                self.basis_vectors = [
                    a * np.array([1.0, 0.0, 0.0]),
                    a * np.array([0.5, np.sqrt(3.0) / 2.0, 0.0]),
                ]

        self.reciprocal_vectors = [
            2
            * np.pi
            * np.matmul(r_mat, self.basis_vectors[1])
            / np.dot(self.basis_vectors[0], np.matmul(r_mat, self.basis_vectors[1])),
            2
            * np.pi
            * np.matmul(r_mat, self.basis_vectors[0])
            / np.dot(self.basis_vectors[1], np.matmul(r_mat, self.basis_vectors[0])),
        ]

    @classmethod
    def from_file(cls, path_to_configuration_file: str) -> None:
        extension = os.path.splitext(path_to_configuration_file)[-1].lower()
        if extension != ".toml":
            raise ValueError(f"expected a `.toml` file, received a `{extension}` file")

        path_to_configuration_file = Path(".").joinpath(path_to_configuration_file)
        if not os.path.exists(path_to_configuration_file):
            raise ValueError(f"file {path_to_configuration_file} not found")

        parsed_configuration = toml.load(path_to_configuration_file)["Lattice"]

        return cls(
            Symmetry[parsed_configuration["symmetry"]],
            parsed_configuration["lengths"],
            parsed_configuration["background_index"],
            parsed_configuration.get("finite_extent"),
            "coupling" in parsed_configuration,
        )

    def unit_cell_area(self) -> float:
        return np.linalg.norm(np.cross(self.basis_vectors[0], self.basis_vectors[1]))

    def particle_count(self) -> Optional[int]:
        if (finite_extent := self.finite_extent) is not None:
            return finite_extent[0] * finite_extent[1]
        else:
            return None

    def coupled_mode_index(
        self, wavelengths_nm: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        if self.spp_coupling:
            return n_spp(wavelengths_nm)
        else:
            return (
                np.ones_like(wavelengths_nm, dtype=np.complex128)
                * self.background_index
            )

    def positions_in_cell(self) -> Optional[npt.NDArray[np.float64]]:
        match self.symmetry:
            case Symmetry.Rectangular:
                return None
            case Symmetry.Hexagonal:
                return None
            case Symmetry.Honeycomb:
                return np.array(
                    [self.lengths_nm[0] * np.array([0.5, np.sqrt(3) / 6, 0.0])]
                )
