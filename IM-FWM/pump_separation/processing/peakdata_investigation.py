import os
import sys
import re

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import load_pump_files, process_multiple_datasets
from pump_separation.funcs.processing import (
    sort_multiple_datasets,
    multi_pumpsep_opt_ce,
    get_subfolders,
)
from pump_separation.funcs.plotting import plot_top_n_datasets
import matplotlib.pyplot as plt

# plt.style.use("custom")
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.ion()
plt.rcParams["figure.figsize"] = (20, 11)

data_folder = "../data/pulse/3us_50duty/manual_specs/"
# data_folder = "../data/pulse/3us_25duty/sortby_redshift/"
# data_folder = "../data/CW/"
data_folder = ["../data/CW/2.3W/"]
# data_folder = "../data/CW/2.15W/"
# data_folder = "../data/CW/manual/"
file_type = "pkl"
data_folders = ["../data/CW/2.3W/1/", "../data/CW/2.15W/", "../data/CW/2W/1/"]
data_folders = ["../data/CW/2.3W/1/", "../data/CW/2W/1/"]
powers = []

for folder in data_folders:
    match = re.search(r"\d+(\.\d+)?", folder)
    if match:
        number = float(match.group())
        powers.append(number)
(
    unique_pairs,
    raw_data,
    sorted_peak_data,
    blue_sorted_peak_data,
    red_sorted_peak_data,
) = sort_multiple_datasets(data_folders, file_type=file_type)

# |%%--%%| <1FvH3TRUP5|2BoWz93aQn>
# # Plot the top n datasets
# meas_num = 0
# n = 1  # Top n datasets
# idx = 0
# plot_top_n_datasets(
#     red_sorted_peak_data[meas_num][unique_pairs[meas_num][idx]],
#     raw_data[meas_num][idx],
#     n,
#     unique_pairs[meas_num][idx],
# )
# # save_plot(
# #     f"./figs/pulsed/CE_top_{n}_datasets_{unique_pairs[idx][0]}_{unique_pairs[idx][1]}_weird_peak"
# # )
# # |%%--%%| <lGX6ibyli3|N6SM5b714x>
# multi_ce_sorted = multi_pumpsep_opt_ce(red_sorted_peak_data, unique_pairs)
# idx = 0
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(
#     multi_ce_sorted[idx][0, :],
#     np.array(multi_ce_sorted[idx][1, :]),
#     marker="o",
#     linestyle="-",
# )
# ax[0].set_ylabel("CE [dB]")
# ax[0].grid(True)
# ax[1].plot(
#     multi_ce_sorted[idx][0, :], multi_ce_sorted[idx][2, :], marker="o", linestyle="-"
# )
# ax[1].set_xlabel("Pump separation [nm]")
# ax[1].set_ylabel("Signal location [nm]")
# ax[1].grid(True)

# fig, ax = plt.subplots()
# for i in range(len(multi_ce_sorted)):
#     ax.plot(
#         multi_ce_sorted[i][0, :],
#         np.array(multi_ce_sorted[i][1, :]),
#         marker="o",
#         linestyle="-",
#         label=f"{powers[i]} W",
#     )
# ax.set_xlabel("Pump separation [nm]")
# ax.set_ylabel("CE [dB]")
# ax.legend()
# ax.grid(True)

# |%%--%%| <N6SM5b714x|K0YrMeUaMb>
# Comparing batches of measurements. The 26 were taken at different times
parent_folder_paths = [
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group1/",
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group2/",
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group3/",
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group4/",
]
multi_ce_sorted_groups = {}
for i, folder_paths in enumerate(parent_folder_paths):
    d = get_subfolders(folder_paths)
    (
        unique_pairs,
        raw_data,
        sorted_peak_data,
        blue_sorted_peak_data,
        red_sorted_peak_data,
    ) = sort_multiple_datasets(d, file_type=file_type)
    multi_ce_sorted_groups[i] = multi_pumpsep_opt_ce(red_sorted_peak_data, unique_pairs)
x_axis = multi_ce_sorted_groups[0][0][0, :]
fig_path = "../figs/CW/vary_pump_p/2.3W/"
#|%%--%%| <K0YrMeUaMb|BrPFyS7DHD>

fig, ax = plt.subplots(2, 1)
for key, multi_ce_sorted_sub in multi_ce_sorted_groups.items():
    num = 0
    for i in range(len(multi_ce_sorted_sub)):
        if num == 0:
            ax[0].plot(
                multi_ce_sorted_sub[i][0, :],
                np.array(multi_ce_sorted_sub[i][1, :]),
                marker="o",
                linestyle="-",
                color=f"{cols[key]}",
                label=f"{key}",
            )
        else:
            ax[0].plot(
                multi_ce_sorted_sub[i][0, :],
                np.array(multi_ce_sorted_sub[i][1, :]),
                marker="o",
                linestyle="-",
                color=f"{cols[key]}",
            )
        ax[1].plot(
            multi_ce_sorted_sub[i][0, :],
            np.array(multi_ce_sorted_sub[i][2, :]),
            marker="o",
            linestyle="-",
            color=f"{cols[key]}",
        )
        num += 1
ax[1].set_xlabel("Pump separation [nm]")
ax[0].set_ylabel("CE [dB]")
ax[1].set_ylabel("Signal wl [nm]")
fig.suptitle("Same measurements, grouped by different times")
ax[0].legend()
ax[0].grid(True)
ax[1].grid(True)
fig.savefig(f"{fig_path}all_raw_multi_ce_sorted_groups.pdf", bbox_inches="tight")
#|%%--%%| <3n66pdMSNZ|VNOkmYZl2O>
mean_ce_groups = {}
std_ce_groups = {}
mean_wl_groups = {}
std_wl_groups = {}
for key, multi_ce_sorted_sub in multi_ce_sorted_groups.items():
    mean_ce_groups[key] = np.mean(np.array(multi_ce_sorted_sub)[:, 1, :], axis=0)
    std_ce_groups[key] = np.std(np.array(multi_ce_sorted_sub)[:, 1, :], axis=0)
    mean_wl_groups[key] = np.mean(np.array(multi_ce_sorted_sub)[:, -1, :], axis=0)
    std_wl_groups[key] = np.std(np.array(multi_ce_sorted_sub)[:, -1, :], axis=0)
# data_dict = {"raw_data": multi_ce_sorted, "mean": mean, "std": std}
# with open("multi_ce_sorted.pkl", "wb") as f:
#     pickle.dump(data_dict, f)
fig, ax = plt.subplots(2, 1)
for key in mean_ce_groups.keys():
    ax[0].plot(x_axis, mean_ce_groups[key], color=f"{cols[key]}", label=f"{key}")
    ax[0].errorbar(
        x_axis, mean_ce_groups[key], yerr=std_ce_groups[key], fmt="o"
    )
    ax[1].plot(x_axis, mean_wl_groups[key], color=f"{cols[key]}")
    ax[1].errorbar(
        x_axis, mean_wl_groups[key], yerr=std_wl_groups[key], fmt="o"
    )
fig.suptitle("Same measurements, grouped by different times")
ax[1].set_xlabel("Pump Separation [nm]")
ax[0].set_ylabel("Mean CE [dB]")
ax[1].set_ylabel("Mean signal wl [nm]")
ax[0].legend()
fig.savefig(f"{fig_path}all_mean_multi_ce_sorted_groups.pdf", bbox_inches="tight")
# |%%--%%| <BrPFyS7DHD|3n66pdMSNZ>
# comparing multiple measurements of same parameters

folder_path = "../data/CW/2.3W/old_power_settings/"
datafolders = get_subfolders(folder_path)
(
    unique_pairs,
    raw_data,
    sorted_peak_data,
    blue_sorted_peak_data,
    red_sorted_peak_data,
) = sort_multiple_datasets(datafolders, file_type=file_type)
multi_ce_sorted = multi_pumpsep_opt_ce(red_sorted_peak_data, unique_pairs)
# |%%--%%| <VNOkmYZl2O|82lKwho1iF>
mean_ce = np.mean(np.array(multi_ce_sorted)[:, 1, :], axis=0)
std_ce = np.std(np.array(multi_ce_sorted)[:, 1, :], axis=0)
mean_wl = np.mean(np.array(multi_ce_sorted)[:, -1, :], axis=0)
std_wl = np.std(np.array(multi_ce_sorted)[:, -1, :], axis=0)
# data_dict = {"raw_data": multi_ce_sorted, "mean": mean, "std": std}
# with open("multi_ce_sorted.pkl", "wb") as f:
#     pickle.dump(data_dict, f)
fig, ax = plt.subplots(2, 1)
ax[0].plot(multi_ce_sorted[0][0, :], mean_ce, label="Mean")
ax[0].errorbar(
    multi_ce_sorted[0][0, :], mean_ce, yerr=std_ce, fmt="ro", label="Standard Deviation"
)
ax[1].plot(multi_ce_sorted[0][0, :], mean_wl, label="Mean")
ax[1].errorbar(
    multi_ce_sorted[0][0, :], mean_wl, yerr=std_wl, fmt="ro", label="Standard Deviation"
)
fig.suptitle("Mean CE and corresponding signal wl for each pump separation")
ax[1].set_xlabel("Pump Separation [nm]")
ax[0].set_ylabel("Mean CE [dB]")
ax[1].set_ylabel("Mean signal wl [nm]")
fig.savefig(f"{fig_path}all_mean_multi_ce_sorted.pdf", bbox_inches="tight")
# |%%--%%| <82lKwho1iF|MZu7GkP1Ji>
fig, ax = plt.subplots(2, 1)
for i in range(len(multi_ce_sorted)):
    ax[0].plot(
        multi_ce_sorted[i][0, :],
        np.array(multi_ce_sorted[i][1, :]),
        marker="o",
        linestyle="-",
        # label=f"{powers[i]} W",
    )
    ax[1].plot(
        multi_ce_sorted[i][0, :],
        np.array(multi_ce_sorted[i][2, :]),
        marker="o",
        linestyle="-",
        # label=f"{powers[i]} W",
    )
ax[1].set_xlabel("Pump separation [nm]")
ax[0].set_ylabel("CE [dB]")
ax[1].set_ylabel("Signal wl [nm]")
ax[0].grid(True)
ax[1].grid(True)
fig.suptitle("CE and corresponding signal wl for each pump separation, all measurements")
fig.savefig(f"{fig_path}all_raw_multi_ce_sorted.pdf", bbox_inches="tight")
# |%%--%%| <MZu7GkP1Ji|bftsCM8xhB>
# Maximum value for each pump sep
max_ce_loc = np.argmax(np.array(multi_ce_sorted)[:, 1, :], axis=0)
max_ce = np.array(multi_ce_sorted)[:, 1, :][max_ce_loc, np.arange(0, 11)]
max_wl = np.array(multi_ce_sorted)[:, 2, :][max_ce_loc, np.arange(0, 11)]
fig, ax = plt.subplots(2, 1)
ax[0].plot(multi_ce_sorted[0][0, :], max_ce)
ax[1].plot(multi_ce_sorted[0][0, :], max_wl)
ax[1].set_xlabel("Pump Separation [nm]")
ax[0].set_ylabel("Max CE [dB]")
ax[1].set_ylabel("Signal wl [nm]")
fig.suptitle("Max CE and corresponding signal wl for each pump separation")
ax[0].grid(True)
ax[1].grid(True)
# fig.savefig(f"{fig_path}max_multi_ce_sorted.pdf", bbox_inches="tight")
#|%%--%%| <bftsCM8xhB|jbtqDP3wOM>

file = np.load("./spectra.npy")
num_iters = file.shape[1]
ce = []
for i in range(num_iters):
    hello = process_single_dataset(file[:, i, :].T, 0.1, -65, (1566, 1576), 0.1, -35)
    ce.append(-hello['differences'][-1])
x_axis = np.arange(-0.25, 0.25, 0.01)
plt.plot(x_axis, ce)
plt.show()
