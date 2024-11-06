import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def rolling_average(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode="same")


plt.style.use("large_fonts")
fig_loc = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs/"
lpg_data_file = "./LP11_TMSI/LPG_T_3_1600_0.6dB/LPGspectra.csv"
lpg_data = np.genfromtxt(lpg_data_file, delimiter=",", skip_header=0)
wls = lpg_data[:, 0]
spectra = lpg_data[:, 1:]
spectra_lin = 10 ** (spectra / 10)
fig, ax = plt.subplots()
ax = cast(Axes, ax)
for i in range(spectra.shape[1]):
    ax.plot(wls, spectra[:, i])

spec_percentage = (1 - spectra_lin[:, -1]) * 100
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
ax.plot(wls, rolling_average(spec_percentage, window=10))
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel(r"$\mathrm{LP}_{01}\rightarrow\mathrm{LP}_{11}$ [\%]")
ax.set_xlim([1565, 1615])
ax.set_ylim([96, 100])
fig.savefig(os.path.join(fig_loc, "LP11_spec.pdf"), bbox_inches="tight")
plt.show()
