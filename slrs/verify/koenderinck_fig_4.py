from slrs.types.calculation import Calculation
import logging
import numpy as np
import toml

logging.basicConfig(level=logging.INFO)

periods_nm = [150.0]
file = "inputs/input_tm_2d_honeycomb.toml"

for period_nm in periods_nm:
    data = toml.load(file)
    data["Lattice"]["lengths"] = [period_nm]
    data["Grid"]["maximum_wavevector_im"] = 4 * np.pi / (period_nm * 1e-9)
    data["Grid"]["minimum_wavevector_im"] = -4 * np.pi / (period_nm * 1e-9)

    probe = data["Field"]["probe"]
    radius_nm = data["Particle"]["radius_nm"]
    volume_m3 = 4 / 3 * np.pi * (radius_nm * 1e-9) ** 3

    f = open(file, "w")
    toml.dump(data, f, encoder=toml.TomlNumpyEncoder())
    f.close()

    calculation = Calculation.from_file(file)
    results = calculation.effective_polarisabilities()

    in_plane_im, wavelengths_nm = calculation.grid.generate()
    unique_wavelengths_nm = wavelengths_nm[:, 0]
    effective_index = unique_wavelengths_nm * 0 + 1
    wavevectors = 2 * np.pi / (1e-9 * unique_wavelengths_nm) * np.real(effective_index)
    scaled_wavevectors = wavevectors / (
        np.pi / (calculation.lattice.lengths_nm[0] * 1e-9)
    )

    xx, yy = calculation.plot_grid()
    xx = xx / (np.pi / (calculation.lattice.lengths_nm[0] * 1e-9))

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mode_wavevectors = 2 * np.pi / (1e-9 * wavelengths_nm) * 1.5

    max_l = 1
    levels = np.linspace(-max_l, max_l, 1000)

    # levels = 100

    fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)

    for ii in range(0, axes.shape[1]):
        col_ax = axes[..., ii]

        plot_value = np.trace(results[ii], axis1=-2, axis2=-1) * (
            2 * 4e-4 * np.pi * mode_wavevectors * period_nm * 1e-9 / volume_m3
        )

        im1 = col_ax[0].contourf(
            xx, yy, np.real(plot_value), levels=levels, extend="both", cmap="seismic"
        )
        im2 = col_ax[1].contourf(
            xx, yy, np.imag(plot_value), levels=levels, extend="both", cmap="seismic"
        )

        for ax in col_ax:
            ax.set_ylabel("Wavenumber (1/cm)")
            ax.set_xlabel("Wavevector ($\pi / a$)")
            ax.set_xlim(-4.0, 4.0)

    divider = make_axes_locatable(axes[-1, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax, orientation="vertical")

    plt.savefig(f"results/infinite_{period_nm}nm_fig_4.png")
    # plt.show()
