# THIS IS THE MAIN DOCUMENT FOR
# C PLUS L BAND SWEEPS
import os
import pickle

import matplotlib.pyplot as plt
from curlyBrace import curlyBrace
import numpy as np
from matplotlib.ticker import MaxNLocator
from pump_separation.funcs.process_multiple_spectra_sorted_in_dicts import (
    process_ce_data_for_pump_sweep_around_opt,
)

plt.style.use("large_fonts")
# plt.rcParams["figure.figsize"] = (16, 11)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()

data_loc_c_sweep = "../data/sweep_multiple_separations_w_polopt/pol_opt_auto/mean_p_wl_1570nm/c_pump_sweep/first_session/merged_data.pkl"
data_loc_l_sweep = "../data/sweep_multiple_separations_w_polopt/pol_opt_auto/mean_p_wl_1570nm/l_pump_sweep/first_session/merged_data.pkl"
with open(data_loc_c_sweep, "rb") as f:
    data_c_sweep = pickle.load(f)
with open(data_loc_l_sweep, "rb") as f:
    data_l_sweep = pickle.load(f)

fig_folder = "../../figs/sweep_multiple_separations_w_polopt/cleo_us_2023/pol_opt_auto"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

save_figs = False
save_spectra = False
merge_data = True

(
    exp_params_c,
    ce_dict_c,
    sig_wl_dict_c,
    idler_wl_dict_c,
    ce_dict_processed_c,
    sig_wl_dict_processed_c,
    idler_wl_dict_processed_c,
    ce_dict_std_c,
    ce_dict_best_for_each_ando_sweep_c,
    ce_dict_best_loc_for_each_ando_sweep_c,
) = process_ce_data_for_pump_sweep_around_opt(data_c_sweep, np.mean)

(
    exp_params_l,
    ce_dict_l,
    sig_wl_dict_l,
    idler_wl_dict_l,
    ce_dict_processed_l,
    sig_wl_dict_processed_l,
    idler_wl_dict_processed_l,
    ce_dict_std_l,
    ce_dict_best_for_each_ando_sweep_l,
    ce_dict_best_loc_for_each_ando_sweep_l,
) = process_ce_data_for_pump_sweep_around_opt(data_l_sweep, np.mean)


def merge_c_l_dicts(
    ce_dict_c,
    ce_dict_l,
    ce_dict_best_loc_for_each_ando_sweep_c,
    ce_dict_best_loc_for_each_ando_sweep_l,
    ando1_wls,
    ando2_wls,
    duty_cycles,
    pump_wl_pairs,
    data_process_method=np.mean,
):
    ce_dict = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_processed = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_processed = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_std = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    best_ce_merged = {
        pump_wl_pair: {dc: [] for dc in duty_cycles} for pump_wl_pair in pump_wl_pairs
    }
    max_ce_vs_pumpsep = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
    std_ce_vs_pumpsep = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        for dc_idx, dc in enumerate(duty_cycles):
            best_ce_slice_c = np.squeeze(ce_dict_c[pump_wl_pair][dc])[
                :, ce_dict_best_loc_for_each_ando_sweep_c[pump_wl_pair][dc]
            ]
            best_ce_slice_l = np.squeeze(ce_dict_l[pump_wl_pair][dc])[
                :, ce_dict_best_loc_for_each_ando_sweep_l[pump_wl_pair][dc]
            ]
            best_ce_slice_merged_tmp = np.concatenate(
                (best_ce_slice_c, best_ce_slice_l), axis=0
            )
            best_ce_merged[pump_wl_pair][dc] = best_ce_slice_merged_tmp

            ce_dict[pump_wl_pair][dc] = np.concatenate(
                (
                    np.squeeze(ce_dict_c[pump_wl_pair][dc]),
                    np.squeeze(ce_dict_l[pump_wl_pair][dc]),
                ),
                axis=0,
            )
            ce_dict_processed[pump_wl_pair][dc] = data_process_method(
                best_ce_slice_merged_tmp, axis=0
            )
            ce_dict_std[pump_wl_pair][dc] = np.std(best_ce_slice_merged_tmp, axis=0)
            max_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = (
                best_ce_slice_merged_tmp.mean()
            )
            max_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = (
                best_ce_slice_merged_tmp.mean()
            )
            std_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = best_ce_slice_merged_tmp.std()
    return (
        ce_dict,
        ce_dict_processed,
        ce_dict_std,
        best_ce_merged,
        max_ce_vs_pumpsep,
        std_ce_vs_pumpsep,
    )


if merge_data:
    (
        ce_dict_merged,
        ce_dict_processed_merged,
        ce_dict_std_merged,
        best_ce_merged,
        max_ce_vs_pumpsep_merged,
        std_ce_vs_pumpsep_merged,
    ) = merge_c_l_dicts(
        ce_dict_c,
        ce_dict_l,
        ce_dict_best_loc_for_each_ando_sweep_c,
        ce_dict_best_loc_for_each_ando_sweep_l,
        exp_params_c["ando1_wls"],
        exp_params_c["ando2_wls"],
        exp_params_c["duty_cycles"],
        exp_params_c["pump_wl_pairs"],
    )


def find_best_ce_and_std_for_each_pump_sep(
    ce_dict_processed,
    ce_dict_std,
    sig_wl_dict_processed,
    idler_wl_dict_processed,
    pump_wl_pairs,
    duty_cycles,
):
    max_ce_vs_pumpsep = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
    std_ce_vs_pumpsep = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
    sig_wl_at_max_ce = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
    idler_wl_at_max_ce = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
    for dc_idx, dc in enumerate(duty_cycles):
        for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
            max_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = np.max(
                ce_dict_processed[pump_wl_pair][dc]
            )
            std_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = np.squeeze(
                ce_dict_std[pump_wl_pair][dc]
            )[np.argmax(ce_dict_processed[pump_wl_pair][dc])]
            sig_wl_at_max_ce[dc_idx, pump_wl_pair_idx] = np.squeeze(
                sig_wl_dict_processed[pump_wl_pair][dc]
            )[np.argmax(ce_dict_processed[pump_wl_pair][dc])]
            idler_wl_at_max_ce[dc_idx, pump_wl_pair_idx] = np.squeeze(
                idler_wl_dict_processed[pump_wl_pair][dc]
            )[np.argmax(ce_dict_processed[pump_wl_pair][dc])]
    sig_wl_at_max_ce = np.mean(sig_wl_at_max_ce, axis=0)
    idler_wl_at_max_ce = np.mean(idler_wl_at_max_ce, axis=0)
    return max_ce_vs_pumpsep, std_ce_vs_pumpsep, sig_wl_at_max_ce, idler_wl_at_max_ce


(
    max_ce_vs_pumpsep_c,
    std_ce_vs_pumpsep_c,
    sig_wl_at_max_ce_c,
    idler_wl_at_max_ce_c,
) = find_best_ce_and_std_for_each_pump_sep(
    ce_dict_processed_c,
    ce_dict_std_c,
    sig_wl_dict_processed_c,
    idler_wl_dict_processed_c,
    exp_params_c["pump_wl_pairs"],
    exp_params_c["duty_cycles"],
)
(
    max_ce_vs_pumpsep_l,
    std_ce_vs_pumpsep_l,
    sig_wl_at_max_ce_l,
    idler_wl_at_max_ce_l,
) = find_best_ce_and_std_for_each_pump_sep(
    ce_dict_processed_l,
    ce_dict_std_l,
    sig_wl_dict_processed_l,
    idler_wl_dict_processed_l,
    exp_params_l["pump_wl_pairs"],
    exp_params_l["duty_cycles"],
)


# |%%--%%| <IaYbPUBNPK|fLqEHhVHkN>
def plot_3d_ce_data(
    pump_sep_ax,
    ando1_wls_rel,
    ando2_wls_rel,
    ce_dict_processed,
    dc_local,
    pump_wl_pairs,
):
    ando1_wls_rel_len = len(ando1_wls_rel[0])
    ando2_wls_rel_len = len(ando2_wls_rel[0])
    if ando1_wls_rel_len > ando2_wls_rel_len:
        y = ando1_wls_rel
        ylabel = r"$\Delta\lambda_{q}$ (nm)"
    else:
        y = ando2_wls_rel
        ylabel = r"$\Delta\lambda_{p}$ (nm)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for pump_sep_idx, pump_sep in enumerate(pump_sep_ax):
        z = np.squeeze(ce_dict_processed[pump_wl_pairs[pump_sep_idx]][dc_local])
        x = np.ones_like(z) * pump_sep
        y_tmp = y[pump_sep_idx]
        ax.plot(x, y_tmp, z, label=f"{pump_sep:.2f} nm")
    ax.set_xlabel(r"$\lambda_q-\lambda_p$ (nm)", labelpad=50)
    ax.set_ylabel(ylabel, labelpad=50)
    ax.set_zlabel("CE (dB)", labelpad=50)
    plt.show()


plot_3d_ce_data(
    exp_params_c["pump_sep_ax"],
    exp_params_c["ando1_wls_rel"],
    exp_params_c["ando2_wls_rel"],
    ce_dict_processed_c,
    0.5,
    exp_params_c["pump_wl_pairs"],
)
plot_3d_ce_data(
    exp_params_l["pump_sep_ax"],
    exp_params_l["ando1_wls_rel"],
    exp_params_l["ando2_wls_rel"],
    ce_dict_processed_l,
    0.5,
    exp_params_l["pump_wl_pairs"],
)


# |%%--%%| <ddsjNlq70o|sebHdZnB0g>
def wl_nm_sep_to_thz(wl1, wl2, c=299792458):
    wl1 = wl1 * 1e-9
    wl2 = wl2 * 1e-9
    freq1 = c / wl1
    freq2 = c / wl2
    thz = np.abs(freq1 - freq2) / 1e12
    return thz


plt.rcParams["figure.figsize"] = (16, 12)
plt.ioff()
dc_plot = [0.1, 0.2, 0.5, 1]


def plot_ce_vs_pump_sep(
    pump_sep_ax,
    max_ce_vs_pumpsep,
    std_ce_vs_pumpsep,
    sig_wl_at_max,
    idler_wl_at_max,
    duty_cycles,
    colors,
    second_ax="thz",
):
    ce_offset = 10 * np.log10(np.array(duty_cycles))
    z_order = np.arange(len(duty_cycles))[::-1]
    fig, ax = plt.subplots()
    for dc_idx, dc in enumerate(duty_cycles):
        if dc == 1:
            dc = "CW"
        ax.plot(
            pump_sep_ax,
            max_ce_vs_pumpsep[dc_idx, :] - ce_offset[dc_idx],
            "o-",
            label=dc,
            color=colors[dc_idx],
            zorder=z_order[dc_idx],
        )
        ax.errorbar(
            pump_sep_ax,
            max_ce_vs_pumpsep[dc_idx, :] - ce_offset[dc_idx],
            yerr=std_ce_vs_pumpsep[dc_idx, :],
            fmt="none",
            ecolor=colors[dc_idx],
            capsize=6,
            capthick=2,
            zorder=z_order[dc_idx],
        )
    ax2 = ax.twiny()
    if second_ax == "freq":
        sig_idler_sep = np.abs(sig_wl_at_max - idler_wl_at_max)
        ax2.plot(sig_idler_sep, max_ce_vs_pumpsep[0, :])
        ax2.set_xlabel(r"$\lambda_i-\lambda_s$ (nm)", labelpad=10)
    elif second_ax == "thz":
        sig_idler_sep_thz = wl_nm_sep_to_thz(sig_wl_at_max, idler_wl_at_max)
        ax2.plot(sig_idler_sep_thz, max_ce_vs_pumpsep[0, :], "o-")
        ax2.set_xlabel(r"$\Delta\omega$ (THz)")
    ax2.get_lines()[0].set_visible(False)
    ax2.grid(False)
    ax.set_xlabel(r"$\lambda_q-\lambda_p$ (nm)")
    ax.set_ylabel(r"CE (dB)")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(title="Duty Cycle", fontsize=28)
    return fig, ax


fig_c, ax_c = plot_ce_vs_pump_sep(
    exp_params_c["pump_sep_ax"],
    max_ce_vs_pumpsep_c,
    std_ce_vs_pumpsep_c,
    sig_wl_at_max_ce_c,
    idler_wl_at_max_ce_c,
    dc_plot,
    colors,
)
fig_l, ax_l = plot_ce_vs_pump_sep(
    exp_params_l["pump_sep_ax"],
    max_ce_vs_pumpsep_l,
    std_ce_vs_pumpsep_l,
    sig_wl_at_max_ce_l,
    idler_wl_at_max_ce_l,
    dc_plot,
    colors,
)
fig_l.tight_layout()
if merge_data:
    fig_merged, ax_merged = plot_ce_vs_pump_sep(
        exp_params_c["pump_sep_ax"],
        max_ce_vs_pumpsep_merged,
        std_ce_vs_pumpsep_merged,
        sig_wl_at_max_ce_c,
        idler_wl_at_max_ce_c,
        dc_plot,
        colors,
    )
if save_figs:
    fig_c.savefig(os.path.join(fig_folder, "ce_vs_pump_sep_c.pdf"), bbox_inches="tight")
    fig_l.savefig(os.path.join(fig_folder, "ce_vs_pump_sep_l.pdf"), bbox_inches="tight")
    # fig_l.savefig(
    #     "/home/thjalfeu/OneDrive/PhD/Projects/papers/cleo_us_2023/figs/ce_vs_pump_sep_l.pdf",
    #     bbox_inches="tight",
    # )
    fig_l.savefig(
        "../../../../../papers/cleo_us_2023/figs/ce_vs_pump_sep_l.pdf",
        bbox_inches="tight",
    )
    if merge_data:
        fig_merged.savefig(
            os.path.join(fig_folder, "ce_vs_pump_sep_merged.pdf"), bbox_inches="tight"
        )


# |%%--%%| <sebHdZnB0g|E6oDyL8hLi>
########### CLEO US 2024 PRESENTATION ############
plt.style.use("large_fonts")


def lin_regress_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    x_return, y_return = np.sort(x), slope * np.sort(x) + intercept
    return x_return, y_return


def plot_ce_vs_pump_sep_cleo_us2024(
    pump_sep_ax,
    max_ce_vs_pumpsep,
    std_ce_vs_pumpsep,
    sig_wl_at_max,
    idler_wl_at_max,
    duty_cycles,
    colors,
    second_ax="thz",
    num_last_points_to_ignore=0,
    xlims=[13, 75],
    ylims=[-37, -4.5],
    lin_regress_last_idx=-2,
):
    end_idx = None if num_last_points_to_ignore == 0 else -num_last_points_to_ignore
    ce_offset = 10 * np.log10(np.array(duty_cycles))
    z_order = np.arange(len(duty_cycles))[::-1] + 5
    fig, ax = plt.subplots(figsize=(14, 10))
    for dc_idx, dc in enumerate(duty_cycles):
        if dc == 1:
            dc = "CW"
        ax.plot(
            pump_sep_ax[:end_idx],
            max_ce_vs_pumpsep[dc_idx, :end_idx] - ce_offset[dc_idx],
            "o-",
            label=dc,
            color=colors[dc_idx],
            zorder=z_order[dc_idx],
        )
        ax.errorbar(
            pump_sep_ax[:end_idx],
            max_ce_vs_pumpsep[dc_idx, :end_idx] - ce_offset[dc_idx],
            yerr=std_ce_vs_pumpsep[dc_idx, :end_idx],
            fmt="none",
            ecolor=colors[dc_idx],
            capsize=6,
            capthick=2,
            zorder=z_order[dc_idx],
        )
        ax.plot(
            *lin_regress_line(
                pump_sep_ax[:lin_regress_last_idx],
                max_ce_vs_pumpsep[dc_idx, :lin_regress_last_idx] - ce_offset[dc_idx],
            ),
            "--",
            color=colors[dc_idx],
            zorder=z_order[dc_idx],
        )
    ax.set_xlim(xlims)
    lower_xlim_freq = wl_nm_sep_to_thz(1570 - xlims[0] / 2, 1570 + xlims[0] / 2)
    upper_xlim_freq = wl_nm_sep_to_thz(1570 - xlims[1] / 2, 1570 + xlims[1] / 2)
    ax.set_ylim(ylims)
    ax2 = ax.twiny()
    if second_ax == "freq":
        sig_idler_sep = np.abs(sig_wl_at_max - idler_wl_at_max)
        ax2.plot(sig_idler_sep, max_ce_vs_pumpsep[0, :])
        ax2.set_xlabel(r"$\Delta\lambda$ (nm)", labelpad=10)
    elif second_ax == "thz":
        sig_idler_sep_thz = wl_nm_sep_to_thz(sig_wl_at_max, idler_wl_at_max)
        ax2.plot(sig_idler_sep_thz, max_ce_vs_pumpsep[0, :], "o-")
        ax2.set_xlabel(r"$\Delta\omega$ (THz)")
    ax2.get_lines()[0].set_visible(False)
    ax2.set_xlim([lower_xlim_freq, upper_xlim_freq])
    ax2.grid(False)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.set_xlabel(r"$\Delta\lambda$ (nm)")
    ax.set_ylabel(r"CE (dB)")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.plot([], [], "--", color="black", label="Linear fit")
    handles, labels = ax.get_legend_handles_labels()
    duty_cycle_legend = ax.legend(
        handles[: len(duty_cycles)],
        labels[: len(duty_cycles)],
        title="Duty Cycle",
        loc="lower left",
    )
    ax.add_artist(duty_cycle_legend)  # Add the first legend manually
    linear_fit_legend = ax.legend(
        handles[len(duty_cycles) :],
        labels[len(duty_cycles) :],
        loc="lower left",
        bbox_to_anchor=(0.2, 0.0),
    )
    ax.add_artist(linear_fit_legend)
    return fig, ax


fig_path_cleous_pres = (
    "/home/thjalfe/Documents/PhD/Projects/papers/cleo_us_2024/presentation/figs/results"
)
ignore_last_points = 1
fig_l, ax_l = plot_ce_vs_pump_sep_cleo_us2024(
    exp_params_l["pump_sep_ax"],
    max_ce_vs_pumpsep_merged,
    std_ce_vs_pumpsep_merged,
    sig_wl_at_max_ce_c,
    idler_wl_at_max_ce_c,
    dc_plot,
    colors,
    num_last_points_to_ignore=ignore_last_points,
    xlims=[13, 70],
    ylims=[-37, -3],
    lin_regress_last_idx=-1,
)
# fig_l, ax_l = plot_ce_vs_pump_sep_cleo_us2024(
#     exp_params_l["pump_sep_ax"],
#     max_ce_vs_pumpsep_l,
#     std_ce_vs_pumpsep_l,
#     sig_wl_at_max_ce_l,
#     idler_wl_at_max_ce_l,
#     dc_plot,
#     colors,
#     num_last_points_to_ignore=ignore_last_points,
# )
fig_l.tight_layout()
if ignore_last_points > 0:
    figname = "ce_vs_pump_sep_l_without_last_points_merged_only_69nm.pdf"
else:
    figname = "ce_vs_pump_sep_l_full_spectrum_merged.pdf"
fig_l.savefig(
    os.path.join(fig_path_cleous_pres, figname),
    bbox_inches="tight",
)
# |%%--%%| <E6oDyL8hLi|nNII6cFZbQ>
duty_cycle_arr_local = np.array(exp_params_c["duty_cycles"][::-1])
ce_offset = 10 * np.log10(duty_cycle_arr_local)
inv_duty_cycles = 1 / duty_cycle_arr_local
fig, ax = plt.subplots()
for p_wl_idx, pwl_sep in enumerate(exp_params_c["pump_sep_ax"]):
    ax.plot(
        10 * np.log10(inv_duty_cycles),
        max_ce_vs_pumpsep_c[::-1, p_wl_idx] - ce_offset,
        "o-",
        label=f"{pwl_sep:.2f} nm",
    )
ax.set_xlabel("1 / Duty Cycle (dB)")
ax.set_ylabel("CE (dB)")
ax.legend(title="Pump Separation")
# |%%--%%| <nNII6cFZbQ|h8DS28sJyY>
pump_sep_idx_local = 0
pump_sep_local = exp_params_c["pump_sep_ax"][pump_sep_idx_local]
max_ce_vs_pumpsep_local = max_ce_vs_pumpsep_c[::-1, pump_sep_idx_local]
slope, intercept = np.polyfit(
    10 * np.log10(inv_duty_cycles),
    max_ce_vs_pumpsep_local - ce_offset,
    1,
)
fig, ax = plt.subplots()
ax.plot(
    10 * np.log10(inv_duty_cycles),
    max_ce_vs_pumpsep_local - ce_offset,
    "o-",
)
ax.plot(
    10 * np.log10(inv_duty_cycles),
    slope * 10 * np.log10(inv_duty_cycles) + intercept,
    "--",
    label=f"slope = {slope:.2f}",
)
ax.set_xlabel("1 / Duty Cycle (dB)")
ax.set_ylabel("CE (dB)")
ax.legend()


# |%%--%%| <nNII6cFZbQ|GDHAseigAh>
def ce_increase_rel_to_cw(max_ce_pump_sep_local: np.ndarray, ce_offset: np.ndarray):
    ce_increase = max_ce_pump_sep_local - ce_offset
    ce_increase_rel_to_cw = ce_increase - ce_increase[0]
    return ce_increase_rel_to_cw


pump_sep_ax = exp_params_c["pump_sep_ax"]
ce_increase_rel_to_cw_all_pump_seps = np.zeros((len(pump_sep_ax), len(ce_offset)))
slope_all_pump_seps = np.zeros(len(pump_sep_ax))
slope_between_all_dc = np.zeros((len(pump_sep_ax), len(ce_offset) - 1))
for pump_sep_idx_local, pump_sep_local in enumerate(pump_sep_ax):
    max_ce_vs_pumpsep_local = max_ce_vs_pumpsep_l[::-1, pump_sep_idx_local]
    ce_increase_rel_to_cw_local = ce_increase_rel_to_cw(
        max_ce_vs_pumpsep_local, ce_offset
    )
    slope, intercept = np.polyfit(
        10 * np.log10(inv_duty_cycles),
        ce_increase_rel_to_cw_local,
        1,
    )
    slope_between_all_dc[pump_sep_idx_local, :] = np.diff(
        ce_increase_rel_to_cw_local
    ) / np.diff(10 * np.log10(inv_duty_cycles))
    ce_increase_rel_to_cw_all_pump_seps[pump_sep_idx_local, :] = (
        ce_increase_rel_to_cw_local
    )
    slope_all_pump_seps[pump_sep_idx_local] = slope
mean_ce_increase_rel_to_cw = np.mean(ce_increase_rel_to_cw_all_pump_seps, axis=0)
std_ce_increase_rel_to_cw = np.std(ce_increase_rel_to_cw_all_pump_seps, axis=0)
mean_slope = np.mean(slope_all_pump_seps)
std_slope = np.std(slope_all_pump_seps)
mean_slope_between_all_dc = np.mean(slope_between_all_dc, axis=0)
std_slope_between_all_dc = np.std(slope_between_all_dc, axis=0)
# |%%--%%| <GDHAseigAh|qlJab9gk88>
fig_path = "/home/thjalfe/Documents/PhD/Projects/papers/cleo_us_2024/presentation/figs/setup_method"
plt.style.use("large_fonts")
fig, ax = plt.subplots()
ax.plot(
    inv_duty_cycles,
    mean_ce_increase_rel_to_cw,
    "o-",
    label="Actual CE",
)
ax.errorbar(
    inv_duty_cycles,
    mean_ce_increase_rel_to_cw,
    std_ce_increase_rel_to_cw,
    fmt="none",
    capsize=6,
)
exp_x = np.linspace(1, 10)
exp_y = 10 * np.log10(np.linspace(1, 10) ** 2)
ax.plot(exp_x, exp_y, "--", label=r"Expected, $P^2$")
ax.set_xscale("log")
for slope_segment, slope in enumerate(mean_slope_between_all_dc):
    mid_x = (inv_duty_cycles[slope_segment] + inv_duty_cycles[slope_segment + 1]) / 2
    mid_y = (
        mean_ce_increase_rel_to_cw[slope_segment]
        + mean_ce_increase_rel_to_cw[slope_segment + 1]
    ) / 2
    ax.text(
        mid_x,
        mid_y,
        # rf"{slope:.2f}$\pm${std_slope_between_all_dc[slope_segment]:.2f}",
        rf"{slope:.2f}",
        fontsize=32,
        ha="center",
        va="top",
        rotation=30,
    )
ax.set_xlabel("1 / Duty Cycle")
ax.legend(fontsize=32)
ax.set_xticks(inv_duty_cycles)
ax.set_xticklabels(1 / duty_cycle_arr_local)
ax.set_ylabel("Relative CE (dB)")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax.set_xlim([np.min(inv_duty_cycles) - 0.01, np.max(inv_duty_cycles) + 0.2])
ax.set_ylim([-0.1, 20.1])
fig.tight_layout()
# fig.savefig(os.path.join(fig_path, "ce_vs_dc_and_slopes.pdf"), bbox_inches="tight")
# |%%--%%| <qlJab9gk88|NyyuKf2Lrg>
# slope between each point
slopes = np.diff(max_ce_vs_pumpsep_local - ce_offset) / np.diff(
    10 * np.log10(inv_duty_cycles)
)
# |%%--%%| <NyyuKf2Lrg|EPNtbLuJZS>
### Plotting spectra
plt.rcParams["figure.figsize"] = (16, 6)
plt.ioff()
pump_spec_loc = "../data/sweep_multiple_separations_w_polopt/cleo/pump_spectra.pkl"
pump_spec_loc = (
    "../data/sweep_multiple_separations_w_polopt/pump_spectra_cplus_l/spectra.pkl"
)
with open(pump_spec_loc, "rb") as f:
    pump_specs = pickle.load(f)

pump_spec = pump_specs[(1607, 1533)]

with open(data_loc_l_sweep, "rb") as f:
    data = pickle.load(f)
large_sep_low_dc = np.squeeze(data[(1607, 1533)]["spectra"][0.1])
best_loc = 10
meas_num = 1
sub_data = large_sep_low_dc[best_loc, meas_num]
# Create a figure and a grid for subplots
width_ratios = [1.5, 1]
fig, (ax, ax2) = plt.subplots(
    1,
    2,
    sharey=True,
    facecolor="w",
    gridspec_kw={"width_ratios": width_ratios, "wspace": 0.1},
)

# Determine the break point on the x-axis
break_start = np.max(
    sub_data[0, :]
)  # This should be set to where you want the axis to break
break_end = np.min(
    pump_spec[0, :]
)  # This should be set to where you want the axis to resume

# Plot the two parts of the x-axis on the different subplots
ax.plot(sub_data[0, :], sub_data[1, :], label="Sub Data")
ax2.plot(pump_spec[0, :], pump_spec[1, :], label="Pump Spectrum")

# Set the limits for the broken axes
ax.set_xlim(np.min(sub_data[0, :]), break_start)
ax2.set_xlim(break_end, np.max(pump_spec[0, :]))
ax.set_xlim(967, 999.5)
ax2.set_xlim(1515, np.max(pump_spec[0, :]))
ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))

# Hide the spines between ax and ax2
ax.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax.yaxis.tick_left()
ax.spines["top"].set_visible(True)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)
ax.tick_params(labelright="off")
ax2.yaxis.tick_right()


# Add diagonal lines to indicate the break in the axis
def draw_diagonals(ax, direction="lr", d=0.015, ax_width=1):
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
    dx = d * ax_width
    dy = d
    if direction == "lr":  # left-to-right
        ax.plot((1 - dx, 1 + dx), (-dy, +dy), **kwargs)
        ax.plot((1 - dx, 1 + dx), (1 - dy, 1 + dy), **kwargs)
    elif direction == "rl":  # right-to-left
        ax.plot((-dx, +dx), (1 - dy, 1 + dy), **kwargs)
        ax.plot((-dx, +dx), (-dy, +dy), **kwargs)


d = 0.015  # Size of the diagonal lines
draw_diagonals(ax, "lr", d=d, ax_width=width_ratios[1])
draw_diagonals(ax2, "rl", d=d, ax_width=width_ratios[0])

# ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dBm)")
# ax2.set_xlabel("Wavelength (nm)")
fig.text(
    0.5, -0.02, "Wavelength (nm)", ha="center", va="center", fontsize=28 * 1.44
)  # Common x-label

ax.tick_params(labelright=False, left=False)
if save_figs:
    fig.savefig(
        "../../../../../papers/cleo_us_2023/figs/spectra.pdf", bbox_inches="tight"
    )
# |%%--%%| <GLbzX0Hu3H|isLdu8wn0K>
plt.style.use("large_fonts")
fig_loc = "/home/thjalfe/Documents/PhD/Projects/papers/cleo_us_2024/presentation/figs/setup_method/"
pump_wls = list(pump_specs.keys())
pump_wls = [item for item in pump_wls if isinstance(item, tuple)]
pump_seps = pump_specs["pump_sep"]
pump_sep_idxs = [2, -1]
pump_sep_idxs = [-2]
pump_sep_val = np.abs(pump_wls[pump_sep_idxs[0]][1] - pump_wls[pump_sep_idxs[0]][0])
x_start = pump_wls[pump_sep_idxs[0]][-1] - 0.3
x_end = pump_wls[pump_sep_idxs[0]][0] + 0.4
y_height = 0
text_loc_x = (x_start + x_end) / 2
text_loc_y = y_height - 6.5
text = f"Pump separation: {pump_sep_val:.0f} nm"
fig, ax = plt.subplots(figsize=(14, 8))
for pump_sep_idx in pump_sep_idxs:
    pump_wls_tmp = pump_wls[pump_sep_idx]
    pump_sep = pump_seps[pump_sep_idx]
    pump_spec = pump_specs[pump_wls_tmp]
    offset = np.max(pump_spec[1, :])
    ax.plot(pump_spec[0, :], pump_spec[1, :] - offset, label=f"{pump_sep:.0f}")
if len(pump_sep_idxs) > 1:
    ax.legend(title=r"$\Delta\lambda$ (nm)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dB)")
ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
ax.yaxis.set_major_locator(MaxNLocator(nbins=9))
ax.annotate(
    "",
    xy=(x_start, y_height),
    xytext=(x_end, y_height),
    arrowprops=dict(arrowstyle="<->", color="black", linewidth=2),
)
ax.text(text_loc_x, text_loc_y, text, ha="center", fontsize=52)
ax.text(text_loc_x, text_loc_y - 13, r"$P_p+P_q\approx$3 W", ha="center", fontsize=52)
x_start_c_pump = 1563
x_start_l_pump = 1577
offset_y = 35
ax.annotate(
    "",
    xy=(x_start_c_pump, y_height - offset_y),
    xytext=(x_start, y_height - offset_y),
    arrowprops=dict(arrowstyle="<-", color="black", linewidth=2),
)
text_loc_c_band = (x_start_c_pump + x_start) / 2
y_loc_c_l_pump = y_height - offset_y + 3
ax.text(text_loc_c_band, y_loc_c_l_pump, "1563-1535.5", ha="center", fontsize=52)
ax.annotate(
    "",
    xy=(x_start_l_pump, y_height - offset_y),
    xytext=(x_end, y_height - offset_y),
    arrowprops=dict(arrowstyle="<-", color="black", linewidth=2),
)
text_loc_l_band = (x_start_l_pump + x_end) / 2
y_loc_c_l_pump = y_height - offset_y + 5
ax.text(text_loc_l_band, y_loc_c_l_pump, "1577-1604.5", ha="center", fontsize=52)
ax.set_xlim([np.min(pump_spec[0, :]), np.max(pump_spec[0, :])])
ax.set_ylim([-55, 0.5])
fig.tight_layout()
# fig.savefig(os.path.join(fig_loc, f"pump_spectra_69_and_74.pdf"), bbox_inches="tight")
fig.savefig(
    os.path.join(fig_loc, f"pump_spectra_{pump_sep_val}.pdf"), bbox_inches="tight"
)
fig.savefig(
    os.path.join(fig_loc, f"pump_spectra_{pump_sep_val}.svg"), bbox_inches="tight"
)
# |%%--%%| <isLdu8wn0K|ecpHqK3mi6>
# plotting sig + idler spectra
with open(data_loc_c_sweep, "rb") as f:
    data = pickle.load(f)
curly_font = {
    "color": "k",
    "weight": "bold",
    "style": "italic",
    "size": 50,
    "rotation": "horizontal",
}
large_sep_low_dc = np.squeeze(data[(1604.5, 1535.5)]["spectra"][0.1])
large_sep_cw = np.squeeze(data[(1604.5, 1535.5)]["spectra"][1])
best_loc = 6
meas_num = 1
spec = large_sep_low_dc[best_loc, meas_num]
spec_cw = large_sep_cw[9, meas_num]
specs = np.array([spec, spec_cw])
sig_peak_idx_arr = np.argmax(specs[:, 1, :], axis=1)
sig_peak_loc_arr = specs[np.arange(specs.shape[0]), 0, sig_peak_idx_arr]
sig_peak_power_arr = specs[np.arange(specs.shape[0]), 1, sig_peak_idx_arr]
idler_peak_idx_arr = np.argmax(specs[:, 1, sig_peak_idx_arr[0] + 100 :], axis=1)
idler_peak_loc_arr = specs[
    np.arange(specs.shape[0]), 0, idler_peak_idx_arr + sig_peak_idx_arr + 100
]
idler_peak_power_arr = specs[
    np.arange(specs.shape[0]), 1, idler_peak_idx_arr + sig_peak_idx_arr + 100
]
bracket_offset_arr = [0, -9]
leg = [r"0.1", "CW"]
fig, ax = plt.subplots(figsize=(11, 8))
for i, spec in enumerate(specs):
    sig_peak_idx = sig_peak_idx_arr[i]
    sig_peak_loc = sig_peak_loc_arr[i]
    sig_peak_power = sig_peak_power_arr[i]
    idler_peak_idx = idler_peak_idx_arr[i]
    idler_peak_loc = idler_peak_loc_arr[i]
    idler_peak_power = idler_peak_power_arr[i]
    curly_x = idler_peak_loc - 2 + bracket_offset_arr[i]
    curly_end = [curly_x, sig_peak_power]
    curly_start = [curly_x, idler_peak_power]
    curly_str_loc = [
        curly_x - 5,
        (sig_peak_power + idler_peak_power) / 2,
    ]
    horizontal_dash_upper = [
        [sig_peak_loc, idler_peak_loc],
        [sig_peak_power, sig_peak_power],
    ]
    horizontal_dash_lower = [
        [curly_x, idler_peak_loc],
        [idler_peak_power, idler_peak_power],
    ]
    ax.plot(spec[0, :], spec[1, :], label=leg[i])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (dBm)")
    ax.set_xlim([np.min(spec[0, :]), np.max(spec[0, :])])
    curlyBrace(
        fig,
        ax,
        curly_start,
        curly_end,
        0.05,
        bool_auto=True,
        color="black",
        lw=4,
        fontdict=curly_font,
    )
    ax.text(
        curly_str_loc[0],
        curly_str_loc[1],
        f"{(idler_peak_power - sig_peak_power):.1f} dB",
        fontsize=40,
        ha="center",
        va="center",
    )
    ax.plot(horizontal_dash_upper[0], horizontal_dash_upper[1], "k--")
    ax.plot(horizontal_dash_lower[0], horizontal_dash_lower[1], "k--")
# offset legend both in x and y
ax.legend(loc="lower left", bbox_to_anchor=(0.08, 0.1), title="Duty Cycle")
ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
fig.tight_layout()
fig.savefig(os.path.join(fig_loc, "sig_idler_spectra.pdf"), bbox_inches="tight")
