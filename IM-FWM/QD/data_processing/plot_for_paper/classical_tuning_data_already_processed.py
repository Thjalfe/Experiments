import pickle
from glob import glob
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plot_for_paper.ce_data_plotting_funcs import (
    plot_ce_diff_dc_rel_to_cw,
    plot_ce_vs_abs_detuning_w_qd,
    plot_ce_vs_detuning_all_dc,
    plot_ce_vs_detuning_freq_bottom_ax,
    plot_ce_vs_detuning_wl_bottom_ax,
    plot_p1_vs_p2_phasematch,
)

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
remove_col_idx = 8
colors.pop(remove_col_idx)
save_figs = True

data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_w_processed_merged_proper_data/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.68/data.pkl"
data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_w_processed_merged_proper_data/p1-wl=1590.5-1596.25nm_p2-wl=1610.60-1570.69/data.pkl"
qd_data_loc = "../../data/ce_data_from_qd_calc-from-peak.pkl"
extra_data_locs = glob(
    "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_780-fiber-out/**/data_processed.pkl",
    recursive=True,
)
extra_data_locs = glob(
    "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/**/data_processed.pkl",
    recursive=True,
)
extra_data_loc = "../../data/classical-opt/sig-wl_same-as-qd/sweep_both_pumps_w_processed_merged_proper_data/p1-wl=1593.5-1596.0nm_p2-wl=1591.55-1572.90_merged_old_equal_p_powers-on-red-side/data_processed.pkl"

# extra_data_loc = extra_data_locs[13]
# 780 idxs: [1, -3] [1, -1],
# For 780 it is 2d because much of the data is merged data, so [filename, idx]
# 1060 idxs: 13
with open(data_loc, "rb") as f:
    data = pickle.load(f)
with open(qd_data_loc, "rb") as f:
    qd_data = pickle.load(f)
with open(extra_data_loc, "rb") as f:
    extra_data = pickle.load(f)
extra_data_pump_sep_thz = extra_data["processed_data"]["pump_sep_thz"]
extra_data_idler_wls = extra_data["processed_data"]["idler_wls"]
extra_data_ce_mean = extra_data["processed_data"]["ce_peak_mean"][:, 0]
extra_data_ce_mean_lin = extra_data["processed_data"]["ce_peak_mean_lin"][:, 0] * 100


ce_classical_mean = data["processed_data"]["ce_peak_mean"]
ce_classical_std = data["processed_data"]["ce_peak_std"]
ce_classical_mean_lin = data["processed_data"]["ce_peak_mean_lin"] * 100
ce_classical_std_lin = data["processed_data"]["ce_peak_std_lin"] * 100
classical_pump_sep_thz = data["processed_data"]["pump_sep_thz"]
classical_red_idxs = data["processed_data"]["red_idxs"]
classical_blue_idxs = data["processed_data"]["blue_idxs"]
p1_wl = data["processed_data"]["p1_wls"]
p2_wl = data["processed_data"]["p2_wl_max_ce"]
dc_arr = data["processed_data"]["dc_per_dataset"][0]
sig_wls = data["processed_data"]["sig_wls"]
idler_wls = data["processed_data"]["idler_wls"]
dc_percentage = dc_arr * 100
idler_wls_qd = qd_data["idler_wls"]
qd_thz_sep = -qd_data["thz_sep"]
qd_ces_lin = qd_data["ces"]
qd_stds_lin = qd_data["stds"]
qd_ces_lin_bound = qd_ces_lin + qd_stds_lin
qd_ces = 10 * np.log10(qd_ces_lin / 100)
qd_bound = 10 * np.log10(qd_ces_lin_bound / 100)
qd_stds = qd_bound - qd_ces
qd_red_idxs_to_ignore = np.array([0])

show_only_ce_plot = False
if not show_only_ce_plot:

    fig1, ax1 = plot_ce_vs_detuning_all_dc(
        dc_arr, ce_classical_mean, ce_classical_std, classical_pump_sep_thz
    )


if not show_only_ce_plot:

    fig2, ax2 = plot_ce_diff_dc_rel_to_cw(
        dc_arr, ce_classical_mean, classical_pump_sep_thz
    )

    fig3, ax3 = plot_p1_vs_p2_phasematch(p1_wl, p2_wl)


if not show_only_ce_plot:

    fig4, ax4 = plot_ce_vs_abs_detuning_w_qd(
        qd_thz_sep,
        qd_ces_lin,
        qd_stds_lin,
        classical_pump_sep_thz,
        ce_classical_mean_lin,
        colors,
        qd_red_idxs_to_ignore=qd_red_idxs_to_ignore,
    )

    fig5, ax5 = plot_ce_vs_detuning_freq_bottom_ax(
        qd_thz_sep,
        qd_ces_lin,
        qd_stds_lin,
        classical_pump_sep_thz,
        ce_classical_mean_lin,
        ce_classical_std_lin,
        idler_wls,
        colors,
    )
    if save_figs:
        fig5.savefig(f"{paper_dir}/ce_vs_detuning_w-classical.pdf", bbox_inches="tight")

# same as plot above but with wavelength as bottom axis

# fig.tight_layout()
fig6, ax6 = plot_ce_vs_detuning_wl_bottom_ax(
    qd_thz_sep,
    qd_ces_lin,
    qd_stds_lin,
    idler_wls_qd,
    classical_pump_sep_thz,
    ce_classical_mean_lin,
    ce_classical_std_lin,
    idler_wls,
    colors,
    qd_red_idxs_to_ignore=qd_red_idxs_to_ignore,
    show_classical_error_bars=False,
    qd_as_rainbow=True,
    extra_data_idler_wls=extra_data_idler_wls,
    extra_data_ce_mean=extra_data_ce_mean_lin,
)
ax6.set_xlim(np.min(idler_wls) - 0.2, np.max(idler_wls) + 0.1)
if save_figs:
    fig6.savefig(
        f"{paper_dir}/ce_vs_detuning_w-classical_lambda_x_ax.pdf", bbox_inches="tight"
    )

fig, ax1 = plt.subplots()
ax1 = cast(Axes, ax1)
fig = cast(Figure, fig)
ax1.plot(
    p1_wl[::-1],
    ce_classical_mean[::-1, 0],
    "-o",
)
from scipy.interpolate import interp1d

transform_to_p2 = interp1d(p1_wl, p2_wl, fill_value="extrapolate")
transform_to_p1 = interp1d(p2_wl, p1_wl, fill_value="extrapolate")
secax = ax1.secondary_xaxis("top", functions=(transform_to_p2, transform_to_p1))
secax.set_xticks(np.linspace(p2_wl.max(), p2_wl.min(), 5))  # adjust spacing as desired
secax.set_xlabel("p2_wl (nm)")
ax1.axvline(1593.2667, color="black", linestyle="--", alpha=0.7)
