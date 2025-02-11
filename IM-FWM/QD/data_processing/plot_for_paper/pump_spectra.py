from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator


def pump_wls_to_thz_sep(p1wl, p2wl, c=299792458):
    p1wl_thz = c / (p1wl * 10**-9) * 10**-12
    p2wl_thz = c / (p2wl * 10**-9) * 10**-12
    return p2wl_thz - p1wl_thz


plt.style.use("custom")
figsize = (11, 6)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = r"\usepackage{upgreek}\usepackage{amsmath}"
# plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 38
plt.rcParams["ytick.labelsize"] = 38
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"

data_dir = "../../data/"
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
remove_col_idx = 8
colors.pop(remove_col_idx)
save_figs = True

data_loc = "../../data/classical-opt/sig-wl_same-as-qd/pump_spectra/data.pkl"
lpg_loc = "../../../../LPG/LP11_TMSI/LPG_T_3_1600_0.6dB/LPGspec_percentage.csv"
with open(data_loc, "rb") as f:
    data_dict = cast(dict, np.load(f, allow_pickle=True))
lpg_data = np.genfromtxt(lpg_loc, delimiter=",")
roll_window = 5
lpg_data[1, :] = np.convolve(
    lpg_data[1, :], np.ones(roll_window) / roll_window, mode="same"
)
pump_wls = data_dict["pump_wls"]
spectra = data_dict["spectra"]
x_lims = (1565, 1615)
idxs_to_loop_over = [7]
# idxs_to_loop_over = np.arange(0, len(spectra))
fig, ax = plt.subplots()
ax = cast(Axes, ax)
for spec_num in idxs_to_loop_over:
    spec = spectra[spec_num]
    spec_norm = np.copy(spec)
    spec_norm[1, :] = spec[1, :] - np.max(spec[1, :])
    spec_norm[1, spec_norm[1, :] < -70] = np.nan
    ax.plot(spec_norm[0, :], spec_norm[1, :])
    ax.set_xlim(x_lims)
    min_val_in_range = np.min(
        spec_norm[1, (spec_norm[0, :] > 1565) & (spec_norm[0, :] < 1615)]
    )
    ax.set_ylim(min_val_in_range, 1)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Rel. power [dB]")

    fig, ax = plt.subplots()

    ax = cast(Axes, ax)
    fig = cast(Figure, fig)
    spec = spectra[spec_num]
    spec_norm = np.copy(spec)
    spec_norm[1, :] = spec[1, :] - np.max(spec[1, :])
    spec_norm[1, spec_norm[1, :] < -70] = np.nan
    ax.plot(spec_norm[0, :], spec_norm[1, :])
    ax.set_xlim(x_lims)
    min_val_in_range = np.min(
        spec_norm[1, (spec_norm[0, :] > 1565) & (spec_norm[0, :] < 1615)]
    )
    ax.set_ylim(min_val_in_range, 1)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Rel. power [dB]")
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
if save_figs:
    fig.savefig(f"{paper_dir}/pump_spectra.pdf", bbox_inches="tight")
image1 = mpimg.imread(f"{paper_dir}/mode_images/LP01_inverted_transparent.png")
start_wl_img = 1570
wl_sep_img = 50
arrow_dist = 13
img_height_loc = 91
image1_box = OffsetImage(image1, zoom=0.184 * 1.5)
image1_ab = AnnotationBbox(image1_box, (start_wl_img, img_height_loc), frameon=False)
image2 = mpimg.imread(f"{paper_dir}/mode_images/LP11_inverted_transparent.png")
image2_box = OffsetImage(image2, zoom=1.5 * 1.4)
image2_ab = AnnotationBbox(
    image2_box, (start_wl_img + wl_sep_img, img_height_loc), frameon=False
)
fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
ax.plot(lpg_data[0, :], lpg_data[1, :], color="k")
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel(r"Mode $\eta$ [\%]")
# ax.set_xlim(x_lims)
ax.set_xlim(1540, 1650)
ax.set_ylim(87, 100)
ax.annotate(
    "",
    xy=(start_wl_img + wl_sep_img - arrow_dist, img_height_loc),
    xytext=(start_wl_img + arrow_dist, img_height_loc),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=2.5),
    zorder=20,
)
ax.add_artist(image1_ab)
ax.add_artist(image2_ab)
if save_figs:
    fig.savefig(f"{paper_dir}/lpg_spec.pdf", bbox_inches="tight")
