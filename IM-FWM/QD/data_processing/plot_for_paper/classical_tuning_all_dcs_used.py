import pickle
import seaborn as sns
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
save_figs = False

data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_780-fiber-out/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40/data.pkl"
# data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/p1-wl=1591-1594.75nm_p2-wl=1608.60-1581.80/data.pkl"
# data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_fine_merge/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40/data.pkl"
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
idler_wls_qd = qd_data["idler_wls"]
p1_wl = np.array(p1_wl)[sort_key]
p2_wl = np.array(p2_wl)[sort_key]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
blue_idxs_to_ignore = [4]
classical_blue_idxs_filtered = np.delete(
    classical_blue_idxs,
    np.array(
        [
            len(classical_blue_idxs) - 1 - blue_idx_to_ignore
            for blue_idx_to_ignore in blue_idxs_to_ignore
        ]
    ),
)  # Done in this way because the blue_idxs_to_ignore are in reverse order to fit with the reversed order for fig3, ax3
cols_for_lines = sns.color_palette("dark:salmon_r", n_colors=len(dc_arr) + 1)[::-1]
idxs_to_move_text_vertically = [3, 4, len(dc_arr) - 1]
move_amount = [0.4, -0.6, 0.32]
for dc_idx, dc in enumerate(dc_arr):
    dc_per = dc_percentage[dc_idx]
    ax.plot(
        pump_sep_thz[classical_blue_idxs_filtered],
        np.squeeze(ce_pol_opt_peak_mean[classical_blue_idxs_filtered, dc_idx]),
        "-o",
        color=cols_for_lines[dc_idx],
        markersize=8,
    )
    if dc_idx in idxs_to_move_text_vertically:
        move_vert = move_amount[idxs_to_move_text_vertically.index(dc_idx)]
    else:
        move_vert = 0
    ax.text(
        pump_sep_thz[classical_blue_idxs_filtered][0] + 0.1,
        np.squeeze(ce_pol_opt_peak_mean[classical_blue_idxs_filtered, dc_idx][0])
        + move_vert,
        # f"DC={dc_arr[dc_idx]:.2f}",
        rf"{dc_per:.1f}\%",
        color=cols_for_lines[dc_idx],
        verticalalignment="center",
        fontsize=24,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

ax.text(
    pump_sep_thz[classical_blue_idxs_filtered][0] - 0.05,
    -1,
    r"Duty cycles",
    color="k",
    verticalalignment="center",
    fontsize=24,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
)
count = 0
for pump_sep_idx in range(len(pump_sep_thz[classical_blue_idxs_filtered])):
    ax.axvline(
        pump_sep_thz[classical_blue_idxs_filtered][::-1][pump_sep_idx],
        linestyle="--",
        color=colors[count],
        linewidth=4,
    )
    count += 1
ax.set_xlim(
    pump_sep_thz[classical_blue_idxs_filtered][-1] - 0.04,
    pump_sep_thz[classical_blue_idxs_filtered][0] + 0.44,
)
ax.set_ylim(
    np.min(ce_pol_opt_peak_mean[classical_blue_idxs_filtered, :]) - 1,
    np.max(ce_pol_opt_peak_mean[classical_blue_idxs_filtered, :]) + 0.5,
)
ax.set_xlabel(r"$\Delta\nu$ [THz]")
ax.set_ylabel(r"$\eta_\mathrm{peak}$ [dB]")
if save_figs:
    fig.savefig(f"{paper_dir}/ce_vs_detuning_blue_all_dcs.pdf", bbox_inches="tight")
# |%%--%%| <mqpcwtWAaz|TXt55gQOFe>

ce_diff_between_dc_rel_to_cw = np.array(
    [
        ce_pol_opt_peak_mean[:, dc_idx] - ce_pol_opt_peak_mean[:, -1]
        for dc_idx in range(len(dc_arr))
    ]
)
ce_diff_between_dc_rel_to_cw_blue_only = np.squeeze(
    ce_diff_between_dc_rel_to_cw[:, classical_blue_idxs]
)
# For the two inner detunings, the detuning is so low that going to low DC does not scale properly
ce_diff_between_dc_rel_to_cw_blue_only_filtered = np.copy(
    ce_diff_between_dc_rel_to_cw_blue_only
)
ce_diff_between_dc_rel_to_cw_blue_only_filtered[0, -2:] = np.nan
ce_diff_between_dc_rel_to_cw_mean = np.nanmean(
    ce_diff_between_dc_rel_to_cw_blue_only_filtered, axis=1
)
ce_diff_dc_rel_to_cw_fit = np.polyfit(
    -10 * np.log10(dc_arr), ce_diff_between_dc_rel_to_cw_mean, 1
)
dc_fit = np.linspace(dc_arr[0], dc_arr[-1], 100)
ce_diff_between_dc_rel_to_cw_mean_lin = 10 ** (ce_diff_between_dc_rel_to_cw_mean / 10)
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
ax.plot(-10 * np.log10(dc_arr), ce_diff_between_dc_rel_to_cw_mean, "-o")
ax.plot(
    -10 * np.log10(dc_fit),
    np.polyval(ce_diff_dc_rel_to_cw_fit, -10 * np.log10(dc_fit)),
    "--",
    color="black",
    label=f"Fit: {ce_diff_dc_rel_to_cw_fit[0]:.2f}x + {ce_diff_dc_rel_to_cw_fit[1]:.2f}",
)
ax.set_xlabel("1/DC [dB]")
ax.set_ylabel(r"$\eta_\mathrm{peak}/\eta_\mathrm{cw}$ [dB]")
ax.legend()
if save_figs:
    fig.savefig(
        f"{paper_dir}/ce_vs_dc_diff_blue_mean_all-seps.pdf", bbox_inches="tight"
    )

fig3, ax3 = plt.subplots()
ce_pol_opt_opt_peak_mean_blue = ce_pol_opt_peak_mean[classical_blue_idxs, :]
ce_pol_opt_opt_peak_mean_blue = ce_pol_opt_opt_peak_mean_blue[::-1, :]
idxs_smaller_fit_range = [0, 1]
saturation_idxs = [3, 2]
ax3 = cast(Axes, ax3)
fig3 = cast(Figure, fig3)
col_counter = 0
for i in range(len(pump_sep_thz[classical_blue_idxs])):
    if i in blue_idxs_to_ignore:
        continue
    if i in idxs_smaller_fit_range:
        dc_arr_loc = dc_arr[saturation_idxs[i] :]
        ce_mean_loc = ce_pol_opt_opt_peak_mean_blue[i, saturation_idxs[i] :]
    else:
        dc_arr_loc = dc_arr
        ce_mean_loc = ce_pol_opt_opt_peak_mean_blue[i, :]

    ce_loc_fit, cov = np.polyfit(-10 * np.log10(dc_arr_loc), ce_mean_loc, 1, cov=True)
    err = np.sqrt(np.diag(cov))

    ax3.plot(
        -10 * np.log10(dc_arr),
        ce_pol_opt_opt_peak_mean_blue[i, :],
        "-o",
        color=colors[col_counter],
    )
    ax3.plot(
        -10 * np.log10(dc_fit),
        np.polyval(ce_loc_fit, -10 * np.log10(dc_fit)),
        "--",
        color=colors[col_counter],
        label=rf"{ce_loc_fit[0]:.2f}$\pm${err[0]:.2f}",
    )
    col_counter += 1
ax3.set_xlabel("1/DC [dB]")
ax3.set_ylabel(r"$\eta_\mathrm{peak}$ [dB]")
ax3.legend(title="Slope", fontsize=20, title_fontsize=24)
ax3.set_xlim(np.min(-10 * np.log10(dc_arr)) - 0.2, np.max(-10 * np.log10(dc_arr)) + 0.2)
ax3.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
ax3.set_ylim(
    np.min(ce_pol_opt_peak_mean[classical_blue_idxs_filtered, :]) - 1,
    np.max(ce_pol_opt_peak_mean[classical_blue_idxs_filtered, :]) + 7.5,
)

if save_figs:
    fig3.savefig(f"{paper_dir}/ce_vs_dc_diff_blue_all-seps.pdf", bbox_inches="tight")
# fig, ax = plt.subplots()
# ax = cast(Axes, ax)
# fig = cast(Figure, fig)
# ax.plot(p1_wl, p2_wl, "-o")


# |%%--%%| <TXt55gQOFe|avffDcxZB8>
def thz_to_wls(thz, thz_sep, idler_wls):
    poly_coeffs = np.polyfit(thz_sep, idler_wls, 1)
    poly = np.poly1d(poly_coeffs)
    return poly(thz)


def wls_to_thz(wls, thz_sep, idler_wls):
    poly_coeffs = np.polyfit(idler_wls, thz_sep, 1)
    poly = np.poly1d(poly_coeffs)
    return poly(wls)


blue_idxs_to_ignore = np.arange(0, 3)
# ignore no idxs
blue_idxs_to_ignore = []
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
