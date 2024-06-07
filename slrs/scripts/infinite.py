# Example script to run a calculation for an infinite array

from slrs.types.calculation import Calculation
from slrs.types.lattice import n_spp
from slrs.utils.conversions import nanometre_to_wavenumber
import logging
import numpy as np
import toml

logging.basicConfig(level=logging.INFO)

periods_nm = [8000.0]

for period_nm in periods_nm:
    data = toml.load("inputs/input_tm_2d_sic_inf.toml")
    data["Lattice"]["lengths"] = [period_nm, period_nm]
    data["Grid"]["maximum_wavevector_im"] = 2 * np.pi / (period_nm * 1e-9)

    probe = data["Field"]["probe"]

    f = open("inputs/input_tm_2d_sic.toml", "w")
    toml.dump(data, f, encoder=toml.TomlNumpyEncoder())
    f.close()

    calculation = Calculation.from_file("inputs/input_tm_2d_sic.toml")
    extinction_tm = calculation.cross_sections()

    _, wavelengths_nm = calculation.grid.generate()
    unique_wavelengths_nm = wavelengths_nm[:, 0]
    effective_index = n_spp(unique_wavelengths_nm)
    wavevectors = 2 * np.pi / (1e-9 * unique_wavelengths_nm) * np.real(effective_index)
    scaled_wavevectors = wavevectors / (
        np.pi / (calculation.lattice.lengths_nm[0] * 1e-9)
    )

    xx, yy = calculation.plot_grid()

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # levels = np.linspace(0.0, 200, 100)
    levels = 100

    fig, ax = plt.subplots(figsize=(12, 12))
    xx = xx / (np.pi / (calculation.lattice.lengths_nm[0] * 1e-9))
    im1 = ax.contourf(xx, yy, extinction_tm, levels=levels, extend="both")

    ax.plot(scaled_wavevectors, nanometre_to_wavenumber(unique_wavelengths_nm))
    ax.plot(2.0 - scaled_wavevectors, nanometre_to_wavenumber(unique_wavelengths_nm))
    ax.set_ylabel("Wavenumber (1/cm)")
    ax.set_xlabel("Wavevector ($\pi / a$)")
    ax.set_xlim(0, 2.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")

    plt.savefig(f"results/infinite_{period_nm}nm.png")
    # plt.show()

    np.savez(f"results/infinite_{period_nm}nm", x=xx, y=yy, z=extinction_tm)
