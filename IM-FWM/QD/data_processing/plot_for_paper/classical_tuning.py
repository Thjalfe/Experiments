import pickle
from typing import cast
from functools import partial
from funcs.processing import find_idler_loc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from funcs.utils import sort_by_ascending_x_ax

plt.style.use("custom")
figsize = (11, 7)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{upgreek}\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{courier}\usepackage{helvet}"
)
# plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 38
plt.rcParams["ytick.labelsize"] = 38
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"

data_dir = "../../data/"
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
save_figs = False

data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_780-fiber-out/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40/data.pkl"
# data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_780-fiber-out/test_new_data_only/data.pkl"
# data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_780-fiber-out/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40_2/data.pkl"
qd_data_loc = "../../data/ce_data_from_qd_calc-from-peak.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)
with open(qd_data_loc, "rb") as f:
    qd_data = pickle.load(f)


def pump_wls_to_thz_sep(p1wl, p2wl, c=299792458):
    p1wl_thz = c / (p1wl * 10**-9) * 10**-12
    p2wl_thz = c / (p2wl * 10**-9) * 10**-12
    return p2wl_thz - p1wl_thz


dc_arr = data["duty_cycle"]
dc_percentage = dc_arr * 100
pump_sep_thz = []
ce_pol_opt_peak = []
p1_wl = []
p2_wl = []
idler_wls = []
sig_wl = []
for p1_tmp in data.keys():
    if type(p1_tmp) is not np.float64:
        continue
    if "ce_peak_pol_opt" not in data[p1_tmp].keys():
        continue
    pump_sep_thz.append(
        pump_wls_to_thz_sep(
            p1_tmp,
            data[p1_tmp]["p2_max_ce"],
        )
    )
    ce_pol_opt_peak.append(data[p1_tmp]["ce_peak_pol_opt"])
    p1_wl.append(p1_tmp)
    p2_wl.append(np.round(data[p1_tmp]["p2_max_ce"], 2))
    idx_tmp = np.argmax(data[p1_tmp]["spectra_pol_opt"][:, :, 1, :], axis=2)
    sig_wl.append(data[p1_tmp]["spectra_pol_opt"][0, 0, 0, idx_tmp])
    _, idler_wl_tmp, _ = find_idler_loc(
        data[p1_tmp]["spectra_pol_opt"][
            0, 0, 1, :
        ],  # dimensions [dc, statistics, wl/powers, spectrum]
        data[p1_tmp]["spectra_pol_opt"][0, 0, 0, :],
        (float(p1_tmp), p2_wl[-1]),
        min_counts_idler=-50,
    )

    idler_wls.append(idler_wl_tmp)
idler_wls = np.array(idler_wls)
pump_sep_thz = np.array(pump_sep_thz)
ce_pol_opt_peak = np.array(ce_pol_opt_peak)
sort_key = np.argsort(pump_sep_thz)
ce_pol_opt_peak = np.array(ce_pol_opt_peak)[sort_key]
ce_pol_opt_peak_lin = 10 ** (ce_pol_opt_peak / 10)
ce_pol_opt_peak_mean = np.mean(ce_pol_opt_peak, axis=2)
ce_pol_opt_peak_std = np.std(ce_pol_opt_peak, axis=2)
ce_pol_opt_peak_mean_lin = np.mean(ce_pol_opt_peak_lin, axis=2)
ce_pol_opt_peak_std_lin = np.std(ce_pol_opt_peak_lin, axis=2)
pump_sep_thz = -np.array(pump_sep_thz)[sort_key]
classical_red_idxs = np.where(pump_sep_thz < 0)[0]
classical_blue_idxs = np.where(pump_sep_thz > 0)[0]
# qd_data["thz_sep"] = -np.array(qd_data["thz_sep"])
idler_wls_qd = qd_data["idler_wls"]
p1_wl = np.array(p1_wl)[sort_key]
p2_wl = np.array(p2_wl)[sort_key]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
for dc_idx, dc in enumerate(dc_arr):
    dc_per = dc_percentage[dc_idx]
    ax.errorbar(
        pump_sep_thz,
        ce_pol_opt_peak_mean[:, dc_idx],
        yerr=ce_pol_opt_peak_std[:, dc_idx],
        label=rf"DC={dc_per:.1f}\%",
        fmt="-o",
    )
ax.set_xlabel("Pump separation (THz)")
ax.set_ylabel("CE peak (pol opt)")
ax.legend()

ce_diff_between_dc_rel_to_cw = np.array(
    [
        ce_pol_opt_peak_mean[:, dc_idx] - ce_pol_opt_peak_mean[:, -1]
        for dc_idx in range(len(dc_arr))
    ]
)
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
for p1_wl_idx, p1_wl_tmp in enumerate(p1_wl):
    ax.plot(
        -10 * np.log10(dc_arr),
        ce_diff_between_dc_rel_to_cw[:, p1_wl_idx],
        "-o",
        label=f"{pump_sep_thz[p1_wl_idx]:.2f} THz",
    )
    ax.plot(-10 * np.log10(dc_arr), -20 * np.log10(dc_arr), "--", color="black")
ax.set_xlabel("1 / DC")
ax.set_ylabel("CE peak difference rel to CW")
ax.legend(title="Pump separation [THz]")

fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
ax.plot(p1_wl, p2_wl, "-o")


# |%%--%%| <IUtztqQ5RY|avffDcxZB8>
def thz_to_wls(thz, thz_sep, idler_wls):
    poly_coeffs = np.polyfit(thz_sep, idler_wls, 1)
    poly = np.poly1d(poly_coeffs)
    return poly(thz)


def wls_to_thz(wls, thz_sep, idler_wls):
    poly_coeffs = np.polyfit(idler_wls, thz_sep, 1)
    poly = np.poly1d(poly_coeffs)
    return poly(wls)


idxs_to_ignore = np.arange(0, 3)
# ignore no idxs
idxs_to_ignore = []
thz_sep_abs = np.abs(qd_data["thz_sep"])
thz_sep_abs_sorted, ces_sorted, stds_sorted = sort_by_ascending_x_ax(
    thz_sep_abs, qd_data["ces"], qd_data["stds"]
)
blue_idxs_qd = np.where(qd_data["thz_sep"] < 0)
red_idxs_qd = np.where(qd_data["thz_sep"] > 0)
red_idxs_qd = red_idxs_qd[0][2:]
blue_ces = qd_data["ces"][blue_idxs_qd]
blue_stds = qd_data["stds"][blue_idxs_qd]
red_ces = qd_data["ces"][red_idxs_qd]
red_stds = qd_data["stds"][red_idxs_qd]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)

ax.errorbar(
    np.abs(pump_sep_thz),
    ce_pol_opt_peak_mean_lin[:, 0] * 100,
    yerr=ce_pol_opt_peak_std_lin[:, 0] * 100,
    label=r"Classical",
    capsize=8,
    fmt="o",
    color="k",
)
ax.errorbar(
    np.abs(qd_data["thz_sep"][blue_idxs_qd]),
    blue_ces,
    yerr=blue_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    label="Blue-shifted idler",
)
ax.errorbar(
    np.abs(qd_data["thz_sep"][red_idxs_qd]),
    red_ces,
    yerr=red_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[0],
    label="Red-shifted idler",
)

ax.set_xlabel(r"$\nu_\mathrm{i}-\nu_\mathrm{s}$ [THz]")
ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
ax.legend()
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)

ax.errorbar(
    pump_sep_thz[classical_red_idxs],
    np.squeeze(ce_pol_opt_peak_mean_lin[classical_red_idxs, 0]) * 100,
    yerr=np.squeeze(ce_pol_opt_peak_std_lin[classical_red_idxs, 0]) * 100,
    capsize=8,
    fmt="--o",
    color="grey",
)
ax.errorbar(
    pump_sep_thz[classical_blue_idxs],
    np.squeeze(ce_pol_opt_peak_mean_lin[classical_blue_idxs, 0]) * 100,
    yerr=np.squeeze(ce_pol_opt_peak_std_lin[classical_blue_idxs, 0] * 100),
    capsize=8,
    label=r"Classical",
    fmt="--o",
    color="grey",
)
ax.errorbar(
    -qd_data["thz_sep"][blue_idxs_qd],
    blue_ces,
    yerr=blue_stds,
    fmt="o",
    markersize=8,
    color=colors[0],
    capsize=8,
    label="Quantum",
)
ax.errorbar(
    -qd_data["thz_sep"][red_idxs_qd],
    red_ces,
    yerr=red_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[0],
    # label="Red-shifted idler",
)
wl_lims = (np.min(idler_wls) - 1, np.max(idler_wls) + 1)
thz_lims = (
    wls_to_thz(wl_lims[1], pump_sep_thz, idler_wls),
    wls_to_thz(wl_lims[0], pump_sep_thz, idler_wls),
)
ax.set_xlim(thz_lims)
ax.set_ylim(0, 110)
ax.set_xlabel(r"$\nu_\mathrm{i}-\nu_\mathrm{s}$ [THz]")
ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
secax = ax.secondary_xaxis(
    "top",
    functions=(
        partial(thz_to_wls, thz_sep=pump_sep_thz, idler_wls=idler_wls),
        partial(wls_to_thz, thz_sep=pump_sep_thz, idler_wls=idler_wls),
    ),
)
secax.set_xlabel(r"$\lambda_\mathrm{i}$ [nm]", labelpad=10)
secax.set_xticks(np.arange(965, 981, 5))
# secax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
secax.set_xlim(
    thz_to_wls(thz_lims[0], pump_sep_thz, idler_wls),
    thz_to_wls(thz_lims[-1], pump_sep_thz, idler_wls),
)

ax.text(
    -0.18,
    1.08,
    r"\textbf{b}",
    transform=ax.transAxes,
    fontsize=48,
    va="top",
    ha="left",
)
ax.legend()
if save_figs:
    fig.savefig(f"{paper_dir}/ce_vs_detuning_w-classical.pdf", bbox_inches="tight")

# same as plot above but with wavelength as bottom axis
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)

ax.errorbar(
    idler_wls[classical_red_idxs],
    np.squeeze(ce_pol_opt_peak_mean_lin[classical_red_idxs, 0]) * 100,
    yerr=np.squeeze(ce_pol_opt_peak_std_lin[classical_red_idxs, 0]) * 100,
    capsize=8,
    fmt="--o",
    color="grey",
)
ax.errorbar(
    idler_wls[classical_blue_idxs],
    np.squeeze(ce_pol_opt_peak_mean_lin[classical_blue_idxs, 0]) * 100,
    yerr=np.squeeze(ce_pol_opt_peak_std_lin[classical_blue_idxs, 0] * 100),
    capsize=8,
    label=r"Classical",
    fmt="--o",
    color="grey",
)
ax.errorbar(
    idler_wls_qd[blue_idxs_qd],
    blue_ces,
    yerr=blue_stds,
    fmt="o",
    markersize=8,
    color=colors[0],
    capsize=8,
    label="Quantum",
)
ax.errorbar(
    idler_wls_qd[red_idxs_qd],
    red_ces,
    yerr=red_stds,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[0],
    # label="Red-shifted idler",
)
ax.axvline(
    thz_to_wls(0, pump_sep_thz, idler_wls), color="black", linestyle="--", alpha=0.7
)
xlims = (np.min(idler_wls) - 1, np.max(idler_wls) + 1)
ax.set_xlim(xlims)
ax.set_ylim(0, 110)
ax.set_xlabel(r"$\lambda_\mathrm{i}$ [nm]")
ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
secax = ax.secondary_xaxis(
    "top",
    functions=(
        partial(wls_to_thz, thz_sep=pump_sep_thz, idler_wls=idler_wls),
        partial(thz_to_wls, thz_sep=pump_sep_thz, idler_wls=idler_wls),
    ),
)
secax.set_xlabel(r"$\nu_\mathrm{i}-\nu_\mathrm{s}$ [THz]", labelpad=10)
secax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
# ax.text(
#     -0.18,
#     1.08,
#     r"\textbf{b}",
#     transform=ax.transAxes,
#     fontsize=48,
#     va="top",
#     ha="left",
# )
ax.legend()
# fig.tight_layout()
if save_figs:
    fig.savefig(
        f"{paper_dir}/ce_vs_detuning_w-classical_lambda_x_ax.pdf", bbox_inches="tight"
    )
# |%%--%%| <avffDcxZB8|lrRR7v4Pv2>
#
fig, ax1 = plt.subplots()
ax1 = cast(Axes, ax1)
fig = cast(Figure, fig)
ax1.plot(
    p1_wl[::-1],
    ce_pol_opt_peak_mean[::-1, 0],
    "-o",
)
from scipy.interpolate import interp1d

transform_to_p2 = interp1d(p1_wl, p2_wl, fill_value="extrapolate")
transform_to_p1 = interp1d(p2_wl, p1_wl, fill_value="extrapolate")
secax = ax1.secondary_xaxis("top", functions=(transform_to_p2, transform_to_p1))
secax.set_xticks(np.linspace(p2_wl.max(), p2_wl.min(), 5))  # adjust spacing as desired
secax.set_xlabel("p2_wl (nm)")
ax1.axvline(1593.2667, color="black", linestyle="--", alpha=0.7)
plt.show()
# |%%--%%| <lrRR7v4Pv2|aEVVbdVrGu>
dc_arr_reduced = [dc_arr[0]]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
for dc_idx, dc in enumerate(dc_arr_reduced):
    dc_per = dc_percentage[dc_idx]
    ax.errorbar(
        pump_sep_thz,
        ce_pol_opt_peak_mean[:, dc_idx],
        yerr=ce_pol_opt_peak_std[:, dc_idx],
        fmt="-o",
        color="k",
    )

ax.errorbar(
    qd_data["thz_sep"][blue_idxs_qd],
    10 * np.log10(blue_ces / 100),
    yerr=blue_stds / blue_ces * 10,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[0],
    label="Blue-shifted idler",
)
ax.errorbar(
    qd_data["thz_sep"][red_idxs_qd],
    10 * np.log10(red_ces / 100),
    yerr=red_stds / red_ces * 10,
    fmt="o",
    markersize=8,
    capsize=8,
    color=colors[3],
    label="Red-shifted idler",
)
ax.set_xlabel(r"$\Delta\nu$ [THz]")
ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
thz_lims = (-3, 3)
ax.set_xlim(thz_lims)
ax.set_xlabel(r"$\Delta\nu$ [THz]")
ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
secax = ax.secondary_xaxis(
    "top",
    functions=(
        partial(thz_to_wls, thz_sep=pump_sep_thz, idler_wls=idler_wls),
        partial(wls_to_thz, thz_sep=pump_sep_thz, idler_wls=idler_wls),
    ),
)
secax.set_xlabel(r"$\lambda_\mathrm{i}$ [nm]", labelpad=10)
secax.set_xticks(np.arange(965, 981, 5))
# secax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
secax.set_xlim(
    thz_to_wls(thz_lims[0], pump_sep_thz, idler_wls),
    thz_to_wls(thz_lims[-1], pump_sep_thz, idler_wls),
)
ax.legend()
plt.show()
# |%%--%%| <aEVVbdVrGu|UkqmbihQsY>
max_std_idx = np.argmax(ce_pol_opt_peak_std_lin[:, 0])
spectra_max_std = data[list(data.keys())[max_std_idx]]["spectra_pol_opt"][0, :, :, :]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
for i in range(spectra_max_std.shape[0]):
    ax.plot(spectra_max_std[i, 0, :], spectra_max_std[i, 1, :])
plt.show()
