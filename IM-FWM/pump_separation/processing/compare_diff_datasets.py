import os
import pickle
import sys
import matplotlib.lines as mlines

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import load_pump_files
from pump_separation.funcs.processing import (
    sort_multiple_datasets,
    multi_pumpsep_opt_ce,
    get_subfolders,
)
from pump_separation.funcs.plotting import plot_top_n_datasets
import matplotlib.pyplot as plt

plt.style.use("custom")
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()
plt.rcParams["figure.figsize"] = (20, 11)


file_type = "pkl"
# |%%--%%| <N6SM5b714x|K0YrMeUaMb>
processed_data_nopolopt = "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/multi_ce_sorted_groups_mean_std.pkl"
processed_data_polopt = "../data/CW/2.3W/pol_dependence/Optimized_pol_for_all_pump_seps/multi_ce_sorted_groups_mean_std.pkl"
with open(processed_data_nopolopt, "rb") as f:
    no_pol_opt = pickle.load(f)
with open(processed_data_polopt, "rb") as f:
    pol_opt = pickle.load(f)

fig_path = "../figs/CW/compare_pol_optimizing/"
save_figs = False
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
x_axis = np.arange(1, 12)
# |%%--%%| <K0YrMeUaMb|S7E52JP6Ln>
# plot mean and std of CE and wl
no_pol_keys_wanted = [0, 3]
fig, ax = plt.subplots(2, 1)
key_num = 0
for key in no_pol_opt["mean_ce_groups"].keys():
    if key in no_pol_keys_wanted:
        ax[0].plot(
            x_axis,
            no_pol_opt["mean_ce_groups"][key],
            color=f"{cols[key_num]}",
            linestyle="dashed",
        )
        ax[0].errorbar(
            x_axis,
            no_pol_opt["mean_ce_groups"][key],
            yerr=no_pol_opt["std_ce_groups"][key],
            fmt="o",
        )
        ax[1].plot(
            x_axis,
            no_pol_opt["mean_wl_groups"][key],
            color=f"{cols[key_num]}",
            linestyle="dashed",
        )
        ax[1].errorbar(
            x_axis,
            no_pol_opt["mean_wl_groups"][key],
            yerr=no_pol_opt["std_wl_groups"][key],
            fmt="o",
        )
        key_num += 1
    else:
        continue
for key in pol_opt["mean_ce_groups"].keys():
    ax[0].plot(
        x_axis,
        pol_opt["mean_ce_groups"][key],
        color=f"{cols[key_num]}",
    )
    ax[0].errorbar(
        x_axis,
        pol_opt["mean_ce_groups"][key],
        yerr=pol_opt["std_ce_groups"][key],
        fmt="o",
    )
    ax[1].plot(
        x_axis,
        pol_opt["mean_wl_groups"][key],
        color=f"{cols[key_num]}",
    )
    ax[1].errorbar(
        x_axis,
        pol_opt["mean_wl_groups"][key],
        yerr=pol_opt["std_wl_groups"][key],
        fmt="o",
    )
    key_num += 1
dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='Pol optimized for 1 nm pump separation')
solid_line = mlines.Line2D([], [], color='black', label='Pol optimized for all pump separations')
ax[0].add_line(dashed_line)
ax[0].add_line(solid_line)
ax[0].legend(handles=[dashed_line, solid_line])

fig.suptitle("Mean and std of CE and signal wl with and without pol optimization")
ax[1].set_xlabel("Pump Separation [nm]")
ax[0].set_ylabel("Mean CE [dB]")
ax[1].set_ylabel("Mean signal wl [nm]")
ax[0].legend()
if save_figs:
    fig.savefig(f"{fig_path}all_mean_sorted_groups.pdf", bbox_inches="tight")
