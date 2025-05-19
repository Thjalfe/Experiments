import pickle
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from funcs.load_data import get_data_filenames, load_data_as_dict
from funcs.processing import (
    calc_multiple_ces_from_peak_values,
    find_idler_loc,
    calc_ce_from_peak_values,
    find_multiple_idler_locs,
    mean_std_data,
)
from funcs.utils import get_exp_fit, pump_wls_to_thz_sep, sort_by_ascending_x_ax
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

plt.style.use("custom")
figsize = (11, 8)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{upgreek}\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{courier}\usepackage{helvet}"
)
save_figs = False
# plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 38
plt.rcParams["ytick.labelsize"] = 38
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"

data_dir = "../../data/"
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore

ref_file_names, data_file_names = get_data_filenames(data_dir)
ref_data, duty_cycles = load_data_as_dict(ref_file_names)
ref_data_mean_std = mean_std_data(ref_data)
data, duty_cycles = load_data_as_dict(data_file_names)
idler_locs, idler_wls, sig_wls = [], [], []
for k in data.keys():
    tmp_idler_locs, tmp_idler_wls, tmp_sig_wls = [], [], []
    dat = data[k]
    wl_ax = dat[0]
    for i in range(1, len(dat)):
        idler_loc, idler_wl, sig_wl = find_idler_loc(dat[i], wl_ax, k)
        tmp_idler_locs.append(idler_loc)
        tmp_idler_wls.append(idler_wl)
        tmp_sig_wls.append(sig_wl)
    idler_locs.append(tmp_idler_locs)
    idler_wls.append(tmp_idler_wls)
    sig_wls.append(tmp_sig_wls)
idler_locs = np.array(idler_locs)
idler_wls = np.array(idler_wls)
sig_wls = np.array(sig_wls)
ces = []
count = 0
for k in data.keys():
    dat = data[k]
    ref = ref_data[k]
    ces_tmp = [
        calc_ce_from_peak_values(
            dat[i + 1],
            ref_data_mean_std[k]["mean"],
            idler_locs[count][i],
            duty_cycles[count],
        )
        for i in range(len(dat) - 1)
    ]
    ces.append(ces_tmp)
    count += 1
# |%%--%%| <WzFxNPrdmv|e0VcR4oPQd>
pump_wls_tuple = list(data.keys())
thz_sep = np.array(
    [pump_wls_to_thz_sep(pump_wls_tuple[i]) for i in range(len(pump_wls_tuple))]
)
pump_wls_arr = np.array(pump_wls_tuple)
data_mean_std = mean_std_data(data)
idler_locs, idler_wls, sig_wls = find_multiple_idler_locs(data_mean_std)
sig_wl = float(np.mean(sig_wls))
ces, stds = calc_multiple_ces_from_peak_values(
    data_mean_std, ref_data_mean_std, idler_locs, duty_cycles
)
# |%%--%%| <e0VcR4oPQd|Z9cqH14aWh>
ces = ces * 100
stds = stds * 100
data_to_save = {
    "thz_sep": thz_sep,
    "idler_wls": idler_wls,
    "ces": ces,
    "stds": stds,
    "sig_wl": sig_wl,
}
with open("../../data/ce_data_from_qd_calc-from-peak.pkl", "wb") as f:
    pickle.dump(data_to_save, f)
blue_idxs = np.where(thz_sep < 0)
red_idxs = np.where(thz_sep > 0)
red_idxs = red_idxs[0][2:]
blue_ces = ces[blue_idxs]
blue_stds = stds[blue_idxs]
red_ces = ces[red_idxs]
red_stds = stds[red_idxs]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
ax.errorbar(idler_wls, ces, yerr=stds * 2, fmt="o", markersize=8, capsize=8)
ax.axvline(sig_wl, color="k", linestyle="--")
ax.set_ylim(0, 100)
ax.set_xlabel("Idler Wavelength (nm)")
ax.set_ylabel(r"Peak Conversion Efficiency (\%)")

fig, ax = plt.subplots()
ax = cast(Axes, ax)
ax.errorbar(thz_sep, ces, yerr=stds * 2, fmt="o", markersize=8, capsize=8)
ax.set_ylim(0, 100)
ax.set_xlabel(r"$\nu_\mathrm{s} - \nu_\mathrm{i}$ [THz]")
ax.set_ylabel(r"Peak Conversion Efficiency (\%)")


# |%%--%%| <Z9cqH14aWh|LE60wY9ojZ>
idxs_to_ignore = np.arange(0, 3)
thz_sep_abs = np.abs(thz_sep)
thz_sep_abs_sorted, ces_sorted, stds_sorted = sort_by_ascending_x_ax(
    thz_sep_abs, ces, stds
)


new_x_ax = np.linspace(0, 3, 100)
exp_fit_fn = get_exp_fit(thz_sep_abs_sorted, ces_sorted, new_x_ax, idxs_to_ignore)
new_x_ax_neg = -new_x_ax
exp_fit_fn_neg = np.flip(exp_fit_fn)[::-1]
fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
ax.errorbar(
    np.abs(thz_sep[blue_idxs]),
    blue_ces,
    yerr=blue_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    label="Blue-shifted idler",
)
ax.errorbar(
    np.abs(thz_sep[red_idxs]),
    red_ces,
    yerr=red_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[3],
    label="Red-shifted idler",
)
ax.text(
    -0.18,
    1.08,
    r"\textbf{a}",
    transform=ax.transAxes,
    fontsize=48,
    va="top",
    ha="left",
)
ax.legend()
ax.plot(new_x_ax, exp_fit_fn, "--", color="k")
ax.set_ylim(0, 100)

ax.set_xlabel(r"$|\nu_\mathrm{s}-\nu_\mathrm{i}|$ [THz]")
ax.set_ylabel(r"$\eta_\mathrm{peak}$ [\%]")
ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))
if save_figs:
    fig.savefig(f"{paper_dir}/ce_vs_abs-detuning_w-fit.pdf", bbox_inches="tight")


def thz_to_wls(thz):
    return np.interp(thz, thz_sep, idler_wls)


def wls_to_thz(wls):
    return np.interp(wls, idler_wls, thz_sep)


fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
ax.errorbar(
    thz_sep[blue_idxs],
    blue_ces,
    yerr=blue_stds,
    fmt="o",
    markersize=8,
    capsize=8,
)
ax.errorbar(
    thz_sep[red_idxs],
    red_ces,
    yerr=red_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[3],
)
ax.plot(new_x_ax, exp_fit_fn, "--", color="k")
ax.plot(new_x_ax_neg, exp_fit_fn_neg, "--", color="k")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 100)
ax.set_xlabel(r"$\nu_\mathrm{s}-\nu_\mathrm{i}$ [THz]")
ax.set_ylabel(r"$\eta_\mathrm{peak}$ [\%]")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
secax = ax.secondary_xaxis("top", functions=(thz_to_wls, wls_to_thz))
secax.set_xlabel(r"$\lambda_\mathrm{i}$ [nm]", labelpad=10)
secax.set_xticks(np.arange(965, 981, 5))
# secax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
secax.set_xlim(thz_to_wls(-3), thz_to_wls(3))
ax.text(
    -0.18,
    1.08,
    r"\textbf{a}",
    transform=ax.transAxes,
    fontsize=48,
    va="top",
    ha="left",
)

if save_figs:
    fig.savefig(f"{paper_dir}/ce_vs_detuning_w-fit.pdf", bbox_inches="tight")
plt.show()
# |%%--%%| <POYjOLt0yB|tpv1zrCUU6>
ignore_idxs = [5, 6]
fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
for i, pump_wl in enumerate(pump_wls_tuple):
    if i in ignore_idxs:
        continue
    ax.plot(data_mean_std[pump_wl]["wl"], 10 * np.log10(data_mean_std[pump_wl]["mean"]))
    wl_idx = np.nanargmin(np.abs(data_mean_std[pump_wl]["wl"] - idler_wls[i]))
    wl = data_mean_std[pump_wl]["wl"][wl_idx]
    y_val = 10 * np.log10(data_mean_std[pump_wl]["mean"][wl_idx])

    # Add an arrow annotation at the idler wavelength position
    ax.annotate(
        "",
        xy=(wl, y_val + 10 * 0.2),
        xytext=(wl, y_val + 10 * 1),  # Adjust y positions for arrow and text
        arrowprops=dict(facecolor="black", shrink=0.05),
    )


pump_wl_ref = pump_wls_tuple[7]
ax.plot(
    ref_data_mean_std[pump_wl_ref]["wl"],
    10 * np.log10(ref_data_mean_std[pump_wl_ref]["mean"]),
    color="k",
    label="Reference",
)
ax.set_ylim(1, 50)
ax.set_xlim(np.min(idler_wls) - 1, np.max(idler_wls) + 1)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel(r"log$_{10}$[Counts]")
if save_figs:
    fig.savefig(f"{paper_dir}/raw_spectra_log.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
for i, pump_wl in enumerate(pump_wls_tuple):
    if i in ignore_idxs:
        continue
    ax.plot(data_mean_std[pump_wl]["wl"], data_mean_std[pump_wl]["mean"])
    wl_idx = np.nanargmin(np.abs(data_mean_std[pump_wl]["wl"] - idler_wls[i]))
    wl = data_mean_std[pump_wl]["wl"][wl_idx]
    y_val = data_mean_std[pump_wl]["mean"][wl_idx]

    # Add an arrow annotation at the idler wavelength position
    ax.annotate(
        "",
        xy=(wl, y_val + 500),
        xytext=(wl, y_val + 10000),  # Adjust y positions for arrow and text
        arrowprops=dict(facecolor="black", shrink=100),
    )


pump_wl_ref = pump_wls_tuple[7]
ax.plot(
    ref_data_mean_std[pump_wl_ref]["wl"],
    ref_data_mean_std[pump_wl_ref]["mean"],
    color="k",
    label="Reference",
)
# ax.set_ylim([1, 5])
ax.set_xlim(np.min(idler_wls) - 1, np.max(idler_wls) + 1)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Counts")
if save_figs:
    fig.savefig(f"{paper_dir}/raw_spectra.pdf", bbox_inches="tight")
plt.show()
