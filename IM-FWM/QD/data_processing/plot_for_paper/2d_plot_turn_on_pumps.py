import numpy as np
import matplotlib.pyplot as plt
from typing import cast
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from funcs.load_data import extract_pump_wls_from_file_name
from funcs.utils import apply_butter_lowpass_filter
from matplotlib.ticker import MaxNLocator
import os

plt.style.use("custom")
figsize = (11, 7)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{upgreek}\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{courier}\usepackage{helvet}"
)
save_figs = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 38
plt.rcParams["ytick.labelsize"] = 38
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"
save_figs = True
data_loc = "../../data/20240923_duty-cycle-0.05/time-evolution/1593p8_1590p67_5x30s_map_time-evolution.txt"
wl1, wl2 = extract_pump_wls_from_file_name(data_loc)
data = pd.read_csv(data_loc, sep="\t", header=None).to_numpy()
wls = data[0, 1:]
data = data[1:, 1:]
data[data <= 0] = 1
sig_idxs = np.arange(1086, 1131)
mask = np.ones(data.shape[1], dtype=bool)
mask[sig_idxs] = False

background = np.mean(data[:9, mask], axis=0)

full_background = np.zeros_like(data)
full_background[:, mask] = background
data = data - full_background
data[data <= 0] = np.min(np.abs(data[data > 0]))


def wl_ax_to_thz(wl, c=299792458):
    return c / (wl * 1e-9) / 1e12


def wl_ax_to_thz_diff(wl, center_wl, c=299792458):
    wl = wl * 1e-9
    center_wl = center_wl * 1e-9
    thz = c / wl / 1e12
    center_thz = c / center_wl / 1e12
    return center_thz - thz


def thz_diff_to_wl_ax(thz_diff, center_wl, c=299792458):
    center_wl = center_wl * 1e-9
    center_thz = c / center_wl / 1e12
    thz_diff = thz_diff + center_thz
    wl = c / (thz_diff * 1e12)
    return wl[::-1] * 1e9


x = wls

cutoff_freq = 20
fs = 2048 / (x[-1] - x[0])

filtered_data = apply_butter_lowpass_filter(data, cutoff_freq, fs, axis=1)
data = filtered_data
signal_wl = wls[np.argmax(data[0, :])]
time = np.linspace(0, 1, data.shape[0])
filtered_data = apply_butter_lowpass_filter(data.T, 3, 1 / (time[1] - time[0]), axis=1)
data = filtered_data.T
data[data <= 0] = np.min(np.abs(data[data > 0]))

X, Y = np.meshgrid(wls, time)
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
im = ax.pcolormesh(X, Y, data, cmap="viridis", vmin=0, vmax=200, rasterized=True)
ax.set_xlim(970.5, 973)
secax = ax.secondary_xaxis(
    "top",
    functions=(
        lambda wl: wl_ax_to_thz_diff(wl, signal_wl),  # Wavelength to THz conversion
        lambda thz_diff: -thz_diff_to_wl_ax(
            thz_diff, signal_wl
        ),  # THz to Wavelength conversion
    ),
)
secax.set_xlabel(r"$\nu_\mathrm{s}-\nu$ [THz]", labelpad=10)
ax.grid(False)
ax.set_ylim(0, 1)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Time [a.u.]")
# max 8 integers
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
if save_figs:
    fig.savefig(os.path.join(paper_dir, "2d_time_evolution.pdf"), bbox_inches="tight")
# |%%--%%| <ECFP2flsMZ|62kLRwgeLO>
idler_loc = np.argmax(data[-1, sig_idxs[-1] :]) + sig_idxs[-1]
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 1, data.shape[0]), data[:, idler_loc])
ax.set_xlabel("Time [a.u.]")
ax.set_ylabel("Intensity [a.u.]")
plt.show()
