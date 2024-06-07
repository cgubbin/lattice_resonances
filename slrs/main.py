from pathlib import Path
import logging
import sys
import toml

import numpy as np
from scipy.constants import speed_of_light

from slrs.types.calculation import Calculation
from slrs.types.lattice import n_spp
from slrs.utils.conversions import nanometre_to_wavenumber

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_file = Path(sys.argv[1])

    # The scan we are doing

    array_sizes = np.linspace(5, 10, 2, dtype=np.int32)
    period_nm = float(sys.argv[2])

    for extent in array_sizes:
        data = toml.load(input_file)
        data["Lattice"]["finite_extent"] = [extent, extent]
        data["Lattice"]["lengths"] = [period_nm, period_nm]
        data["Grid"]["maximum_wavevector_im"] = 2 * np.pi / (10e-6)
        data["Grid"]["minimum_wavevector_im"] = -2 * np.pi / (10e-6)

        probe = data["Field"]["probe"]

        f = open(input_file, "w")
        toml.dump(data, f, encoder=toml.TomlNumpyEncoder())
        f.close()

        calculation = Calculation.from_file(input_file)
        extinction_tm = calculation.cross_sections()

        _, wavelengths_nm = calculation.grid.generate()
        unique_wavelengths_nm = wavelengths_nm[:, 0]
        effective_index = n_spp(unique_wavelengths_nm)
        wavevectors = (
            2 * np.pi / (1e-9 * unique_wavelengths_nm) * np.real(effective_index)
        )
        scaled_wavevectors = wavevectors / (
            np.pi / (calculation.lattice.lengths_nm[0] * 1e-9)
        )

        xx, yy = calculation.plot_grid()
        unique_xx = xx[0] * calculation.lattice.background_index
        omega = speed_of_light * unique_xx / calculation.lattice.background_index
        wavelength_nm = speed_of_light / (omega / (2 * np.pi)) / 1e-9
        wavenumber_icm = np.abs(nanometre_to_wavenumber(wavelength_nm))

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # levels = np.linspace(0.0, 200, 100)
        levels = 100

        fig, ax = plt.subplots(figsize=(12, 12))
        xx = xx / (np.pi / (calculation.lattice.lengths_nm[0] * 1e-9))
        unique_xx = unique_xx / (np.pi / (calculation.lattice.lengths_nm[0] * 1e-9))
        im1 = ax.contourf(xx, yy, extinction_tm, levels=levels, extend="both")

        for ii in range(-10, 10):
            ax.plot(
                ii * 2.0 - scaled_wavevectors,
                nanometre_to_wavenumber(unique_wavelengths_nm),
                "k--",
            )
            ax.plot(
                ii * 2.0 + scaled_wavevectors,
                nanometre_to_wavenumber(unique_wavelengths_nm),
                "k--",
            )

        ax.plot(unique_xx, wavenumber_icm)
        ax.set_ylabel("Wavenumber (1/cm)")
        ax.set_xlabel("Wavevector ($\pi / a$)")
        # ax.set_xlim(0, 2.0)
        limit = data["Grid"]["maximum_wavevector_im"] / (
            np.pi / (calculation.lattice.lengths_nm[0] * 1e-9)
        )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(
            data["Grid"]["minimum_wavenumber_icm"],
            data["Grid"]["maximum_wavenumber_icm"],
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")

        plt.savefig(f"results/{period_nm}_{extent}x{extent}_{probe}.png")
        # plt.show()

        np.savez(
            f"results/{period_nm}_{extent}x{extent}_{probe}",
            x=xx,
            y=yy,
            z=extinction_tm,
        )
