from pump_separation.funcs.process_multiple_spectra_sorted_in_dicts import (
    process_ce_data_for_pump_sweep_around_opt,
)
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.style.use("custom")
plt.rcParams["figure.figsize"] = (16, 11)
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
merge_data = False

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
# |%%--%%| <ddsjNlq70o|B63NzxnC7p>
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
    sig_idler_sep = np.abs(sig_wl_at_max - idler_wl_at_max)
    ax2 = ax.twiny()
    ax2.plot(sig_idler_sep, max_ce_vs_pumpsep[0, :])
    ax2.get_lines()[0].set_visible(False)
    ax2.set_xlabel(r"$\lambda_i-\lambda_s$ (nm)", labelpad=10)
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
    # "/home/thjalfeu/OneDrive/PhD/Projects/papers/cleo_us_2023/figs/ce_vs_pump_sep_l.pdf",
    # bbox_inches="tight",
    # )
    fig_l.savefig(
        "../../../../../papers/cleo_us_2023/figs/ce_vs_pump_sep_l.pdf",
        bbox_inches="tight",
    )
    if merge_data:
        fig_merged.savefig(
            os.path.join(fig_folder, "ce_vs_pump_sep_merged.pdf"), bbox_inches="tight"
        )
# |%%--%%| <B63NzxnC7p|nNII6cFZbQ>
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


pump_sep_idx_local = 0
ce_increase_rel_to_cw_all_pump_seps = np.zeros(
    (len(exp_params_c["pump_sep_ax"]), len(ce_offset))
)
slope_all_pump_seps = np.zeros(len(exp_params_c["pump_sep_ax"]))
slope_between_all_dc = np.zeros((len(exp_params_c["pump_sep_ax"]), len(ce_offset) - 1))
for pump_sep_idx_local, pump_sep_local in enumerate(exp_params_c["pump_sep_ax"]):
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
fig, ax = plt.subplots()
ax.plot(
    inv_duty_cycles,
    mean_ce_increase_rel_to_cw,
    "o-",
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
ax.plot(exp_x, exp_y, "--", label="expected")
ax.set_xscale("log")
for slope_segment, slope in enumerate(mean_slope_between_all_dc):
    mid_x = (inv_duty_cycles[slope_segment] + inv_duty_cycles[slope_segment + 1]) / 2
    mid_y = (
        mean_ce_increase_rel_to_cw[slope_segment]
        + mean_ce_increase_rel_to_cw[slope_segment + 1]
    ) / 2
    angle_deg = np.degrees(np.arctan(slope))
    ax.text(
        mid_x,
        mid_y,
        rf"{slope:.2f}$\pm${std_slope_between_all_dc[slope_segment]:.2f}",
        fontsize=32,
        ha="center",
        va="top",
        rotation=angle_deg,
    )
ax.set_xlabel("1 / Duty Cycle relative to CW")
ax.set_xticks(inv_duty_cycles)
ax.set_xticklabels(1 / duty_cycle_arr_local)
ax.set_ylabel("Relative CE (dB)")
plt.show()
# |%%--%%| <qlJab9gk88|NyyuKf2Lrg>
# slope between each point
slopes = np.diff(max_ce_vs_pumpsep_local - ce_offset) / np.diff(
    10 * np.log10(inv_duty_cycles)
)
# |%%--%%| <NyyuKf2Lrg|sUBJEyGu6O>
### Plotting spectra
plt.rcParams["figure.figsize"] = (16, 6)
plt.ioff()
pump_spec_loc = "../../data/sweep_multiple_separations_w_polopt/cleo/pump_spectra.pkl"
with open(pump_spec_loc, "rb") as f:
    pump_spec = pickle.load(f)
pump_spec = pump_spec[(1607, 1533)]

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
