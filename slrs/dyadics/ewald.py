# evaluate the Ewald summation of the Green's dyadic for the infinite array
from dataclasses import dataclass
from itertools import product
from typing import List
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.special as sp


# Spectral component of the Ewald sum
@dataclass
class Spectral:
    @staticmethod
    def alpha(
        reciprocal_wavevectors_inm: npt.NDArray[np.float64],
        in_plane_wavevectors_inm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
    ):
        """
        Defined in Eq. 3.11 of Jordan
        10.1016/0021-9991(86)90093-8

        Equivalent to sqrt( | k_nm + k_in |^2 - k^2) / 2
        """
        return (
            np.sqrt(
                np.linalg.norm(
                    reciprocal_wavevectors_inm + in_plane_wavevectors_inm, axis=-1
                )
                ** 2
                - wavevectors_inm**2
            )
            / 2
        )

    @staticmethod
    def fz(
        alpha_inm: npt.NDArray[np.complex128],
        z_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ):
        beta_p = alpha_inm / eta_inm + eta_inm * z_nm
        beta_m = alpha_inm / eta_inm - eta_inm * z_nm

        return (
            sp.erfc(beta_p) * np.exp(2 * alpha_inm * z_nm)
            + sp.erfc(beta_m) * np.exp(-2.0 * alpha_inm * z_nm)
        ) / alpha_inm

    @staticmethod
    def fz1(
        alpha_inm: npt.NDArray[np.complex128],
        z_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ):
        beta_p = alpha_inm / eta_inm + eta_inm * z_nm
        beta_m = alpha_inm / eta_inm - eta_inm * z_nm

        return 2.0 * (
            sp.erfc(beta_p) * np.exp(2.0 * alpha_inm * z_nm)
            - sp.erfc(beta_m) * np.exp(-2.0 * alpha_inm * z_nm)
        )

    @staticmethod
    def fzz1(
        alpha_inm: npt.NDArray[np.complex128],
        z_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ):
        beta_p = alpha_inm / eta_inm + eta_inm * z_nm
        return 4.0 * alpha_inm**2 * Spectral.fz(
            alpha_inm, z_nm, eta_inm
        ) - 8.0 * eta_inm / np.sqrt(np.pi) * np.exp(
            -(alpha_inm**2 + z_nm**2 * eta_inm**4) / eta_inm**2
        )

    @staticmethod
    def xz(
        beta_inm: npt.NDArray[np.float64],
        gamma_inm: npt.NDArray[np.complex128],
        evaluation_point_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.float64]:
        z_nm = evaluation_point_nm[2]

        df_dz = Spectral.fz1(gamma_inm, z_nm, eta_inm)

        return 1j * beta_inm[..., 0] * df_dz

    @staticmethod
    def yz(
        beta_inm: npt.NDArray[np.float64],
        gamma_inm: npt.NDArray[np.complex128],
        evaluation_point_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.float64]:
        z_nm = evaluation_point_nm[2]

        df_dz = Spectral.fz1(gamma_inm, z_nm, eta_inm)

        return 1j * beta_inm[..., 1] * df_dz

    @staticmethod
    def zz(
        gamma_inm: npt.NDArray[np.complex128],
        evaluation_point_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.float64]:
        z_nm = evaluation_point_nm[2]

        df_dzz = Spectral.fzz1(gamma_inm, z_nm, eta_inm)

        return df_dzz

    @staticmethod
    def single_term(
        wavevectors_inm: npt.NDArray[np.float64],
        beta_inm: npt.NDArray[np.float64],
        gamma_inm: npt.NDArray[np.complex128],
        unit_cell_area_nm2: float,
        evaluation_point_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        betabeta = -np.einsum(
            "...j,...k->...jk", beta_inm, beta_inm, dtype=np.complex128
        )
        for ii in range(3):
            betabeta[..., ii, ii] += wavevectors_inm**2

        z_nm = evaluation_point_nm[2]

        out = (
            betabeta
            * Spectral.fz(gamma_inm, z_nm, eta_inm)[..., np.newaxis, np.newaxis]
        )

        out[..., 0, 2] = Spectral.xz(beta_inm, gamma_inm, evaluation_point_nm, eta_inm)
        out[..., 2, 0] = out[..., 0, 2]
        out[..., 1, 2] = Spectral.yz(beta_inm, gamma_inm, evaluation_point_nm, eta_inm)
        out[..., 2, 1] = out[..., 1, 2]
        out[..., 2, 2] += Spectral.zz(gamma_inm, evaluation_point_nm, eta_inm)

        return (
            out
            * np.exp(1j * np.dot(beta_inm, evaluation_point_nm))[
                ..., np.newaxis, np.newaxis
            ]
            / unit_cell_area_nm2
            / 8.0
        )

    @staticmethod
    def off_diagonal(
        wavevectors_inm: npt.NDArray[np.float64],
        beta_inm: npt.NDArray[np.float64],
        gamma_inm: npt.NDArray[np.complex128],
        unit_cell_area_nm2: float,
        evaluation_point_nm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        z_nm = evaluation_point_nm[2]

        df_dz = Spectral.fz1(gamma_inm, z_nm, eta_inm)

        off_diagonal = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)

        off_diagonal[..., 0, 1] = 1j * df_dz
        off_diagonal[..., 1, 0] = -off_diagonal[..., 0, 1]

        f = Spectral.fz(gamma_inm, z_nm, eta_inm)

        off_diagonal[..., 0, 2] = -f * beta_inm[..., 1]
        off_diagonal[..., 2, 0] = -off_diagonal[..., 0, 2]

        off_diagonal[..., 1, 2] = f * beta_inm[..., 0]
        off_diagonal[..., 2, 1] = -off_diagonal[..., 1, 2]

        return (
            off_diagonal
            * wavevectors_inm[..., np.newaxis, np.newaxis]
            * np.exp(1j * np.dot(beta_inm, evaluation_point_nm))[
                ..., np.newaxis, np.newaxis
            ]
            / unit_cell_area_nm2
            / 8.0
        )

    @staticmethod
    def construct(
        basis_vectors_m: List[npt.NDArray[np.float64]],
        reciprocal_vectors_im: List[npt.NDArray[np.float64]],
        coupled_mode_index: npt.NDArray[np.float64],
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
        unit_cell_area_m2: float,
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
    ):
        """ """
        evaluation_point_nm = np.asarray(evaluation_point_nm)
        evaluation_out_of_plane_nm = evaluation_point_nm[2]

        # convergence parameter eta
        basis_vectors_nm = [basis_vector / 1e-9 for basis_vector in basis_vectors_m]
        reciprocal_vectors_inm = [
            reciprocal_vector * 1e-9 for reciprocal_vector in reciprocal_vectors_im
        ]
        eta_inm = 2 * np.sqrt(np.pi) / sum(np.linalg.norm(a) for a in basis_vectors_nm)

        wavevectors_inm = 2 * np.pi / (wavelengths_nm) * coupled_mode_index
        in_plane_wavevectors_inm = in_plane_wavevectors_im * 1e-9

        unit_cell_area_nm2 = unit_cell_area_m2 / 1e-18

        diagonal = np.zeros((*wavelengths_nm.shape, 3, 3), dtype=np.complex128)
        off_diagonal = np.zeros((*wavelengths_nm.shape, 3, 3), dtype=np.complex128)

        for ii, jj in product(range(-cutoff, cutoff + 1), range(-cutoff, cutoff + 1)):
            position_vector_nm = sum(
                i * a for (i, a) in zip([ii, jj], basis_vectors_nm)
            )
            reciprocal_vector_inm = sum(
                i * b for (i, b) in zip([ii, jj], reciprocal_vectors_inm)
            )

            beta_inm = (
                reciprocal_vector_inm[np.newaxis, np.newaxis, :]
                + in_plane_wavevectors_inm
            )

            alpha_inm = Spectral.alpha(
                reciprocal_vector_inm[np.newaxis, np.newaxis, :],
                in_plane_wavevectors_inm,
                wavevectors_inm,
            )

            diagonal += Spectral.single_term(
                wavevectors_inm,
                beta_inm,
                alpha_inm,
                unit_cell_area_nm2,
                evaluation_point_nm,
                eta_inm,
            )

            off_diagonal += Spectral.off_diagonal(
                wavevectors_inm,
                beta_inm,
                alpha_inm,
                unit_cell_area_nm2,
                evaluation_point_nm,
                eta_inm,
            )

        return np.block(
            [
                [diagonal, off_diagonal],
                [off_diagonal, diagonal],
            ]
        )


# Spatial component of the Ewald sum
@dataclass
class Spatial:
    @staticmethod
    def fRn(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        Computes f(Rn) as defined by eq. 17

        Computes f^2 \rho, related to the quantity in Eq. 52
        """
        beta_p = radius * eta + 1j * wavevectors / (2 * eta)
        beta_m = radius * eta - 1j * wavevectors / (2 * eta)
        return np.exp(-1j * wavevectors * radius) * sp.erfc(beta_m) + np.exp(
            1j * wavevectors * radius
        ) * sp.erfc(beta_p)

    @staticmethod
    def fRnb(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        Computes f(Rn) as defined by eq. 17

        Computes f^2 \rho, related to the quantity in Eq. 52
        """
        beta_p = radius * eta + 1j * wavevectors / (2 * eta)
        beta_m = radius * eta - 1j * wavevectors / (2 * eta)
        return (
            (
                np.exp(-1j * wavevectors * radius) * sp.erfc(beta_m)
                + np.exp(1j * wavevectors * radius) * sp.erfc(beta_p)
            )
            / radius
            / 2
        )

    @staticmethod
    def gRn(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        Computes g(Rn) as defined by eq. 17

        Computes f^2 \rho, related to the quantity in Eq. 52
        """
        beta_p = radius * eta + 1j * wavevectors / (2 * eta)
        beta_m = radius * eta - 1j * wavevectors / (2 * eta)
        return (
            (
                -np.exp(-1j * wavevectors * radius) * sp.erfc(beta_m)
                + np.exp(1j * wavevectors * radius) * sp.erfc(beta_p)
            )
            / radius
            / 2
        )

    @staticmethod
    def aRn(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        Computes g(Rn) as defined by eq. 17

        Computes f^2 \rho, related to the quantity in Eq. 52
        """
        beta_m = radius * eta - 1j * wavevectors / (2 * eta)
        return (
            2
            * eta
            / np.sqrt(np.pi)
            * np.exp(-1j * wavevectors * radius)
            * np.exp(-(beta_m**2))
            / 2
            / radius
        )

    @staticmethod
    def bRn(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        Computes g(Rn) as defined by eq. 17

        Computes f^2 \rho, related to the quantity in Eq. 52
        """
        beta_p = radius * eta + 1j * wavevectors / (2 * eta)
        return (
            2
            * eta
            / np.sqrt(np.pi)
            * np.exp(1j * wavevectors * radius)
            * np.exp(-(beta_p**2))
            / 2
            / radius
        )

    @staticmethod
    def fRn_koen(
        rho_nm: float, wavevectors_inm: npt.NDArray[np.float64], eta_inm: float
    ) -> npt.NDArray[np.complex128]:
        """
        Computes f(Rn) as defined by eq. 17

        Computes f^2 \rho, related to the quantity in Eq. 52
        """
        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)
        return np.exp(-1j * wavevectors_inm * rho_inm) * sp.erfc(beta_m) + np.exp(
            1j * wavevectors_inm * rho_inm
        ) * sp.erfc(beta_p)

    @staticmethod
    def fpRn(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        first derivative of f(Rn) using eq. B1 and B2
        """
        beta_p = radius * eta + 1j * wavevectors / (2 * eta)
        beta_m = radius * eta - 1j * wavevectors / (2 * eta)
        a = (
            2
            * eta
            / np.sqrt(np.pi)
            * np.exp(-1j * wavevectors * radius)
            * np.exp(-(beta_m**2))
        )
        b = (
            2
            * eta
            / np.sqrt(np.pi)
            * np.exp(1j * wavevectors * radius)
            * np.exp(-(beta_p**2))
        )
        return (
            1j
            * wavevectors
            * (
                np.exp(1j * wavevectors * radius) * sp.erfc(beta_p)
                - np.exp(-1j * wavevectors * radius) * sp.erfc(beta_m)
            )
            - a
            - b
        )

    @staticmethod
    def fppRn(
        radius: float, wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """ """
        beta_p = radius * eta + 1j * wavevectors / (2 * eta)
        beta_m = radius * eta - 1j * wavevectors / (2 * eta)
        a = (
            2
            * eta
            / np.sqrt(np.pi)
            * np.exp(-1j * wavevectors * radius)
            * np.exp(-(beta_m**2))
        )
        b = (
            2
            * eta
            / np.sqrt(np.pi)
            * np.exp(1j * wavevectors * radius)
            * np.exp(-(beta_p**2))
        )
        c = (
            4
            * eta**2
            / np.sqrt(np.pi)
            * beta_m
            * np.exp(-1j * wavevectors * radius)
            * np.exp(-(beta_m**2))
        )
        d = (
            4
            * eta**2
            / np.sqrt(np.pi)
            * beta_p
            * np.exp(1j * wavevectors * radius)
            * np.exp(-(beta_p**2))
        )
        return (
            -(wavevectors**2) * Spatial.fRn(radius, wavevectors, eta)
            + 2j * wavevectors * (a - b)
            + c
            + d
        )

    @staticmethod
    def fp0(
        wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        returns first derivative of f(Rn) evaluated at origin using eq. B6 (Appendix)
        """
        return -4 * eta / np.sqrt(np.pi) * np.exp(
            (wavevectors / (2 * eta)) ** 2
        ) + 1j * wavevectors * (
            sp.erfc(1j * wavevectors / (2 * eta))
            - sp.erfc(-1j * wavevectors / (2 * eta))
        )

    @staticmethod
    def fppp0(
        wavevectors: npt.NDArray[np.float64], eta: float
    ) -> npt.NDArray[np.complex128]:
        """
        returns third derivative of f(Rn) evaluated at origin using eq. B6 (Appendix)
        """
        return 8 * eta**3 / np.sqrt(np.pi) * np.exp(
            (wavevectors / (2 * eta)) ** 2
        ) - wavevectors**2 * Spatial.fp0(wavevectors, eta)

    @staticmethod
    def fppX(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns d2f_dx2 = d2f_drho2 (drho_dx)^2 + df_drho d2rho_dx2
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dx = rho_vec_nm[0] / rho_nm
        d2rho_dx2 = (rho_vec_nm[1] ** 2 + rho_vec_nm[2] ** 2) / rho_nm**3

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b
        d2f_drho2 = (
            (2.0 / rho_nm**2 - wavevectors_inm**2) * f
            - 2j * wavevectors_inm * g / rho_nm
            + 2 * (a + b) / rho_nm
            + 2 * eta_inm * (a * beta_m + b * beta_p)
        )

        return d2f_drho2 * (drho_dx) ** 2 + df_drho * d2rho_dx2

    @staticmethod
    def fpXpY(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns d2f_dx2 = d2f_drho2 (drho_dx drho_dy) + df_drho d2rho_dxdy
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dx = rho_vec_nm[0] / rho_nm
        drho_dy = rho_vec_nm[1] / rho_nm
        d2rho_dxdy = -rho_vec_nm[0] * rho_vec_nm[1] / rho_nm**3

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b
        d2f_drho2 = (
            (2.0 / rho_nm**2 - wavevectors_inm**2) * f
            - 2j * wavevectors_inm * g / rho_nm
            + 2 * (a + b) / rho_nm
            + 2 * eta_inm * (a * beta_m + b * beta_p)
        )

        return d2f_drho2 * (drho_dx * drho_dy) + df_drho * d2rho_dxdy

    @staticmethod
    def fpXpZ(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns d2f_dxdz = d2f_drho2 (drho_dx drho_dz) + df_drho d2rho_dxdz
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dx = rho_vec_nm[0] / rho_nm
        drho_dz = rho_vec_nm[2] / rho_nm
        d2rho_dxdz = -rho_vec_nm[0] * rho_vec_nm[2] / rho_nm**3

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b
        d2f_drho2 = (
            (2.0 / rho_nm**2 - wavevectors_inm**2) * f
            - 2j * wavevectors_inm * g / rho_nm
            + 2 * (a + b) / rho_nm
            + 2 * eta_inm * (a * beta_m + b * beta_p)
        )

        return d2f_drho2 * (drho_dx * drho_dz) + df_drho * d2rho_dxdz

    @staticmethod
    def fpYpZ(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns d2f_dydz = d2f_drho2 (drho_dy drho_dz) + df_drho d2rho_dydz
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dy = rho_vec_nm[1] / rho_nm
        drho_dz = rho_vec_nm[2] / rho_nm
        d2rho_dydz = -rho_vec_nm[1] * rho_vec_nm[2] / rho_nm**3

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b
        d2f_drho2 = (
            (2.0 / rho_nm**2 - wavevectors_inm**2) * f
            - 2j * wavevectors_inm * g / rho_nm
            + 2 * (a + b) / rho_nm
            + 2 * eta_inm * (a * beta_m + b * beta_p)
        )

        return d2f_drho2 * (drho_dy * drho_dz) + df_drho * d2rho_dydz

    @staticmethod
    def fppY(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns d2f_dy2 = d2f_drho2 (drho_dy)^2 + df_drho d2rho_dy2
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dy = rho_vec_nm[1] / rho_nm
        d2rho_dy2 = (rho_vec_nm[0] ** 2 + rho_vec_nm[2] ** 2) / rho_nm**3

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b
        d2f_drho2 = (
            (2.0 / rho_nm**2 - wavevectors_inm**2) * f
            - 2j * wavevectors_inm * g / rho_nm
            + 2 * (a + b) / rho_nm
            + 2 * eta_inm * (a * beta_m + b * beta_p)
        )

        return d2f_drho2 * (drho_dy) ** 2 + df_drho * d2rho_dy2

    @staticmethod
    def fppZ(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns d2f_dz2 = d2f_drho2 (drho_dz)^2 + df_drho d2rho_dz2
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dz = rho_vec_nm[2] / rho_nm
        d2rho_dz2 = (rho_vec_nm[0] ** 2 + rho_vec_nm[1] ** 2) / rho_nm**3

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b
        d2f_drho2 = (
            (2.0 / rho_nm**2 - wavevectors_inm**2) * f
            - 2j * wavevectors_inm * g / rho_nm
            + 2 * (a + b) / rho_nm
            + 2 * eta_inm * (a * beta_m + b * beta_p)
        )

        return d2f_drho2 * (drho_dz) ** 2 + df_drho * d2rho_dz2

    @staticmethod
    def fpY(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns df_dy = df_drho (drho_dy)
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dy = rho_vec_nm[1] / rho_nm

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b

        return df_drho * drho_dy

    @staticmethod
    def fpX(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns df_dy = df_drho (drho_dy)
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dx = rho_vec_nm[0] / rho_nm

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b

        return df_drho * drho_dx

    @staticmethod
    def fpZ(
        rho_vec_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Returns df_dz = df_drho (drho_dz)
        """

        rho_nm = np.linalg.norm(rho_vec_nm)

        drho_dz = rho_vec_nm[2] / rho_nm

        beta_p = rho_nm * eta_inm + 1j * wavevectors_inm / (2 * eta_inm)
        beta_m = rho_nm * eta_inm - 1j * wavevectors_inm / (2 * eta_inm)

        a = Spatial.aRn(rho_nm, wavevectors_inm, eta_inm)
        b = Spatial.bRn(rho_nm, wavevectors_inm, eta_inm)
        f = Spatial.fRnb(rho_nm, wavevectors_inm, eta_inm)
        g = Spatial.gRn(rho_nm, wavevectors_inm, eta_inm)

        df_drho = -f / rho_nm + g * 1j * wavevectors_inm - a - b

        return df_drho * drho_dz

    @staticmethod
    def f_matrix_koen(
        position_nm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        # R_n = rn - r0
        out = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)

        rho_vec_nm = position_nm - evaluation_point_nm

        out[..., 0, 0] = Spatial.fppX(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 1, 1] = Spatial.fppY(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 2, 2] = Spatial.fppZ(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 0, 1] = Spatial.fpXpY(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 0, 2] = Spatial.fpXpZ(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 1, 2] = Spatial.fpYpZ(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 1, 0] = out[..., 0, 1]
        out[..., 2, 0] = out[..., 0, 2]
        out[..., 2, 1] = out[..., 1, 2]

        return out

    @staticmethod
    def f_matrix(
        position_nm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        # R_n = rn - r0
        rn_nm = position_nm - evaluation_point_nm

        radius_nm = np.linalg.norm(rn_nm)
        rr_nm = np.outer(rn_nm, rn_nm) / radius_nm**2

        m2 = (
            Spatial.fppRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm
            - 3 * Spatial.fpRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm**2
            + 3 * Spatial.fRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm**3
        )

        return m2[..., np.newaxis, np.newaxis] * rr_nm[np.newaxis, np.newaxis, ...]

    @staticmethod
    def off_diagonal_matrix(
        position_nm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        wavevectors_inm: npt.NDArray[np.float64],
        eta_inm: float,
    ) -> npt.NDArray[np.complex128]:
        # modified_position = modified_position / 1e-9
        # wavevectors_im = wavevectors_im * 1e-9
        # eta = eta * 1e-9

        # radius_nm = np.linalg.norm(modified_position_nm)
        # rr = np.array([
        #     [0.0, modified_position_nm[2], -modified_position_nm[1]],
        #     [-modified_position_nm[2], 0.0, modified_position_nm[0]],
        #     [modified_position_nm[1], -modified_position_nm[0], 0],
        # ]) / radius_nm

        rho_vec_nm = position_nm - evaluation_point_nm

        out = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)

        out[..., 0, 1] = Spatial.fpZ(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 0, 2] = -Spatial.fpY(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 1, 2] = Spatial.fpX(rho_vec_nm, wavevectors_inm, eta_inm)
        out[..., 1, 0] = -out[..., 0, 1]
        out[..., 2, 0] = -out[..., 0, 2]
        out[..., 2, 1] = -out[..., 1, 2]

        # m1 = (
        #     Spatial.fpRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm
        #         - Spatial.fRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm ** 2
        # )

        # out = 1j * (
        #     m1[..., np.newaxis, np.newaxis] * rr[np.newaxis, np.newaxis, ...] * wavevectors_inm[..., np.newaxis, np.newaxis]
        # )

        return out * wavevectors_inm[..., np.newaxis, np.newaxis] * 1j

    @staticmethod
    def scalar(
        basis_vectors_nm: List[npt.NDArray[np.float64]],
        reciprocal_vectors_inm: List[npt.NDArray[np.float64]],
        wavevectors_inm: npt.NDArray[np.float64],
        in_plane_wavevectors_inm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
        eta_inm: float,
        include_origin=False,
    ) -> npt.NDArray[np.complex128]:
        out = np.zeros_like(wavevectors_inm, dtype=np.complex128)

        # loop over lattice sites (excluding origin):
        for ii, jj in product(range(-cutoff, cutoff + 1), range(-cutoff, cutoff + 1)):
            if ii == 0 and jj == 0 and np.all(evaluation_point_nm == 0.0):
                continue
            position_nm = sum(i * a for (i, a) in zip([ii, jj], basis_vectors_nm))
            rn_nm = position_nm - evaluation_point_nm
            radius_nm = np.linalg.norm(rn_nm)

            reciprocal_vector_inm = sum(
                i * b for (i, b) in zip([ii, jj], reciprocal_vectors_inm)
            )
            beta_inm = (
                reciprocal_vector_inm[np.newaxis, np.newaxis, :]
                + in_plane_wavevectors_inm
            )
            # **MODIFIED**: next line was exp(ic*dot(kb, Rvec))  --> exp(ic*dot(kb, Rvec - r))

            m1 = (
                Spatial.fpRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm**2
                - Spatial.fRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm**3
            )

            out += np.exp(1j * np.dot(in_plane_wavevectors_inm, position_nm)) * (
                Spatial.fRnb(radius_nm, wavevectors_inm, eta_inm) * wavevectors_inm**2
                # Spatial.fRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm * wavevectors_inm ** 2
                #   + m1
            )

        if np.all(evaluation_point_nm == 0.0):
            if include_origin:
                # add i=j=0 term that was skipped over above:
                radius_nm = np.linalg.norm(evaluation_point_nm)
                # **MODIFIED**: commented out next line and replaced with version immediately below:
                # G_spatial += exp(ic*dot(kb, r)) * fRn(k, R, E) / (8*pi*R)	# if there's an issue, check +\- signs here for r
                out += Spatial.fRn(radius_nm, wavevectors_inm, eta_inm) / radius_nm
            else:
                # subtract off limit term as described by Capolino:
                out += (
                    Spatial.fp0(wavevectors_inm, eta_inm) * wavevectors_inm**2
                    - 2j * wavevectors_inm**3
                )

        return out / (8 * np.pi)

    @staticmethod
    def tensor(
        basis_vectors_nm: List[npt.NDArray[np.float64]],
        reciprocal_vectors_inm: List[npt.NDArray[np.float64]],
        wavevectors_inm: npt.NDArray[np.float64],
        in_plane_wavevectors_inm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
        eta_inm: float,
        include_origin=False,
    ) -> npt.NDArray[np.complex128]:
        out = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)

        # loop over lattice sites (excluding origin):
        for ii, jj in product(range(-cutoff, cutoff + 1), range(-cutoff, cutoff + 1)):
            if ii == 0 and jj == 0 and np.all(evaluation_point_nm == 0.0):
                continue

            position_nm = sum(i * a for (i, a) in zip([ii, jj], basis_vectors_nm))

            reciprocal_vector_inm = sum(
                i * b for (i, b) in zip([ii, jj], reciprocal_vectors_inm)
            )
            beta_inm = (
                reciprocal_vector_inm[np.newaxis, np.newaxis, :]
                + in_plane_wavevectors_inm
            )

            out += np.exp(1j * np.dot(in_plane_wavevectors_inm, position_nm))[
                ..., np.newaxis, np.newaxis
            ] * Spatial.f_matrix_koen(
                position_nm, evaluation_point_nm, wavevectors_inm, eta_inm
            )

        if np.all(evaluation_point_nm == 0.0):
            if include_origin:
                # todo
                # out += Spatial.f_matrix(wavevectors_inm, radius_nm, eta_inm)
                out += 0
            else:
                for ii in range(3):
                    out[..., ii, ii] += (
                        Spatial.fppp0(wavevectors_inm, eta_inm)
                        + 2j * wavevectors_inm**3
                    ) / 3

        return out / (8 * np.pi)

    @staticmethod
    def off_diagonal_block(
        basis_vectors_nm: List[npt.NDArray[np.float64]],
        reciprocal_vectors_inm: List[npt.NDArray[np.float64]],
        wavevectors_inm: npt.NDArray[np.float64],
        in_plane_wavevectors_inm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
        eta_inm: float,
        include_origin=False,
    ) -> npt.NDArray[np.complex128]:
        """Eq. 29."""
        out = np.zeros((*wavevectors_inm.shape, 3, 3), dtype=np.complex128)

        # loop over lattice sites (excluding origin):
        for ii, jj in product(range(-cutoff, cutoff + 1), range(-cutoff, cutoff + 1)):
            if ii == 0 and jj == 0:
                continue
            position_nm = sum(i * a for (i, a) in zip([ii, jj], basis_vectors_nm))
            modified_r_nm = evaluation_point_nm - position_nm
            radius_nm = np.linalg.norm(modified_r_nm)

            reciprocal_vector_inm = sum(
                i * b for (i, b) in zip([ii, jj], reciprocal_vectors_inm)
            )
            beta_inm = (
                reciprocal_vector_inm[np.newaxis, np.newaxis, :]
                + in_plane_wavevectors_inm
            )

            out += np.exp(1j * np.dot(in_plane_wavevectors_inm, position_nm))[
                ..., np.newaxis, np.newaxis
            ] * Spatial.off_diagonal_matrix(
                position_nm, evaluation_point_nm, wavevectors_inm, eta_inm
            )

        return out / (8 * np.pi)

    @staticmethod
    def construct(
        basis_vectors_m: List[npt.NDArray[np.float64]],
        reciprocal_vectors_im: List[npt.NDArray[np.float64]],
        coupled_mode_index: npt.NDArray[np.float64],
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
        include_origin: bool = False,
    ):
        """ """
        if include_origin and np.linalg.norm(evaluation_point_nm) <= 0.1:
            raise ValueError("evaluating at origin while including the origin point")
        evaluation_point_nm = np.asarray(evaluation_point_nm)
        evaluation_out_of_plane_nm = evaluation_point_nm[2]

        # convergence parameter eta
        basis_vectors_nm = [basis_vector / 1e-9 for basis_vector in basis_vectors_m]
        reciprocal_vectors_inm = [
            reciprocal_vector * 1e-9 for reciprocal_vector in reciprocal_vectors_im
        ]
        eta_inm = 2 * np.sqrt(np.pi) / sum(np.linalg.norm(a) for a in basis_vectors_nm)

        wavevectors_inm = 2 * np.pi / (wavelengths_nm) * coupled_mode_index
        in_plane_wavevectors_inm = in_plane_wavevectors_im * 1e-9

        diagonal_block = np.zeros((*wavelengths_nm.shape, 3, 3), dtype=np.complex128)

        scalar = Spatial.scalar(
            basis_vectors_nm,
            reciprocal_vectors_inm,
            wavevectors_inm,
            in_plane_wavevectors_inm,
            evaluation_point_nm,
            cutoff,
            eta_inm,
            include_origin,
        )

        for ii in range(3):
            diagonal_block[..., ii, ii] += scalar

        diagonal_block += Spatial.tensor(
            basis_vectors_nm,
            reciprocal_vectors_inm,
            wavevectors_inm,
            in_plane_wavevectors_inm,
            evaluation_point_nm,
            cutoff,
            eta_inm,
            include_origin,
        )

        # diagonal_block -= (
        #     Spatial.GD_single(wavevectors_inm, np.zeros(3), evaluation_point_nm + np.array([0.0, 0.0, 0.1]))
        #         + Spatial.GD_single(wavevectors_inm, np.zeros(3), evaluation_point_nm + np.array([0.0, 0.0, -0.1]))
        # ) / 2

        off_diagonal_block = Spatial.off_diagonal_block(
            basis_vectors_nm,
            reciprocal_vectors_inm,
            wavevectors_inm,
            in_plane_wavevectors_inm,
            evaluation_point_nm,
            cutoff,
            eta_inm,
            include_origin,
        )

        return np.block(
            [
                [diagonal_block, off_diagonal_block],
                [off_diagonal_block, diagonal_block],
            ]
        )


@dataclass
class EwaldDyadic:
    @staticmethod
    def _construct(
        basis_vectors: List[npt.NDArray[np.float64]],
        reciprocal_vectors: List[npt.NDArray[np.float64]],
        unit_cell_area: float,
        coupled_mode_index: npt.NDArray[np.float64],
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
        include_origin=False,
    ) -> npt.NDArray[np.complex128]:
        spectral = Spectral.construct(
            basis_vectors,
            reciprocal_vectors,
            coupled_mode_index,
            in_plane_wavevectors_im,
            wavelengths_nm,
            unit_cell_area,
            evaluation_point_nm,
            cutoff,
        )

        spatial = Spatial.construct(
            basis_vectors,
            reciprocal_vectors,
            coupled_mode_index,
            in_plane_wavevectors_im,
            wavelengths_nm,
            evaluation_point_nm,
            cutoff,
            include_origin,
        )

        return (np.conj(spectral) + spatial) * 4 * np.pi / 1e-27

    @staticmethod
    def construct(
        basis_vectors: List[npt.NDArray[np.float64]],
        reciprocal_vectors: List[npt.NDArray[np.float64]],
        unit_cell_area: float,
        positions_in_cell_nm: Optional[npt.NDArray[np.float64]],
        coupled_mode_index: npt.NDArray[np.float64],
        in_plane_wavevectors_im: npt.NDArray[np.float64],
        wavelengths_nm: npt.NDArray[np.float64],
        evaluation_point_nm: npt.NDArray[np.float64],
        cutoff: int,
        include_origin=False,
    ) -> npt.NDArray[np.complex128]:
        on_site = EwaldDyadic._construct(
            basis_vectors,
            reciprocal_vectors,
            unit_cell_area,
            coupled_mode_index,
            in_plane_wavevectors_im,
            wavelengths_nm,
            evaluation_point_nm,
            cutoff,
            include_origin,
        )

        if positions_in_cell_nm is not None:
            positions_in_cell_nm = positions_in_cell_nm[0]
            off_site_21 = EwaldDyadic._construct(
                basis_vectors,
                reciprocal_vectors,
                unit_cell_area,
                coupled_mode_index,
                in_plane_wavevectors_im,
                wavelengths_nm,
                evaluation_point_nm - positions_in_cell_nm,
                cutoff,
                include_origin,
            )

            off_site_12 = EwaldDyadic._construct(
                basis_vectors,
                reciprocal_vectors,
                unit_cell_area,
                coupled_mode_index,
                in_plane_wavevectors_im,
                wavelengths_nm,
                positions_in_cell_nm - evaluation_point_nm,
                cutoff,
                include_origin,
            )
            return np.block([[on_site, off_site_12], [off_site_21, on_site]])
        else:
            return on_site
