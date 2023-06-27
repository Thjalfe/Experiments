mport os
import sys

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import (
    load_raw_data,
    get_all_unique_pairs_list,
    analyze_data,
    get_signal_wavelength,
)
from pump_separation.funcs.processing import (
    get_subfolders,
)
import matplotlib.pyplot as plt

# plt.style.use("custom")
# plt.rcParams["figure.figsize"] = (20, 11)
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()
fig_path = "../figs/CW/vary_pump_p/2.3W/"
# |%%--%%| <kcd7YHU7RJ|FiG2GIYWCd>

single_folder = "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group1/1/"


def sigwl_ce_all_files(data, unique_pairs):
    folder_analyzed = {}
    for wl_pair, pump_wl_pair_dict in data.items():
        temp_array = np.zeros((2, len(unique_pairs)))
        i = 0
        for key, sub_dict in pump_wl_pair_dict.items():
            sig_wl = get_signal_wavelength(sub_dict)
            temp_array[0, i] = sig_wl
            try:
                temp_array[1, i] = np.array(sub_dict["differences"])[
                    sub_dict["peak_positions"] > sig_wl
                ][0]
            except (IndexError, TypeError):
                temp_array[1, i] = np.nan

            i += 1
        folder_analyzed[wl_pair] = temp_array
    return folder_analyzed


def sigwl_ce_all_files_init(folder, sortby="redshift"):
    unique_pairs = get_all_unique_pairs_list(folder)
    all_sorted, blueshift_sorted, redshift_sorted, unsorted = analyze_data(folder, pump_wl_pairs=unique_pairs)
    if sortby == "redshift":
        data = redshift_sorted
    elif sortby == "blueshift":
        data = blueshift_sorted
    sorted_data = sigwl_ce_all_files(data, unique_pairs)
    unsorted_data = sigwl_ce_all_files(unsorted, unique_pairs)
    return sorted_data, unsorted_data, unique_pairs


# |%%--%%| <NlOlmdF5cv|dxVjp7xImj>
parent_folder_paths = [
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group1/",
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group2/",
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group3/",
    "../data/CW/2.3W/old_power_settings/sorted_by_time_groups/group4/",
]
parent_folder_paths = [
    "../data/CW/2.3W/pol_dependence/Optimized_pol_for_all_pump_seps/const_sigstart_every_pumpsep/",
    "../data/CW/2.3W/pol_dependence/Optimized_pol_for_all_pump_seps/random_sigstart_every_pumpsep/",
]
multi_ce_sorted_groups = []
multi_ce_unsorted_groups = []
unique_pairs = []
for i, folder_paths in enumerate(parent_folder_paths):
    d = get_subfolders(folder_paths)
    multi_ce_sorted_groups.append([])
    multi_ce_unsorted_groups.append([])
    for folder in d:
        multi_ce_sorted_groups[i].append(sigwl_ce_all_files_init(folder)[0])
        multi_ce_unsorted_groups[i].append(sigwl_ce_all_files_init(folder)[1])
    unique_pairs.append(sigwl_ce_all_files_init(folder)[2])
# |%%--%%| <G7cT3K0Ebs|9ElgvzA9pU>
idx = 3
# sub_data = multi_ce_unsorted_groups[idx]
for pair in unique_pairs[idx]:
    wl_sep = pair[1] - pair[0]
    fig, ax = plt.subplots()
    for group_num, group in enumerate(multi_ce_unsorted_groups):
        for i, meas in enumerate(group):
            zero_bool = ~np.isclose(meas[pair][1, :], 0, atol=0.1)
            if i == 0:
                plt.plot(meas[pair][0, zero_bool], -meas[pair][1, zero_bool], "-o", color=cols[group_num], label=f"{group_num}")
            else:
                plt.plot(meas[pair][0, zero_bool], -meas[pair][1, zero_bool], "-o", color=cols[group_num])
    ax.set_xlabel("Signal wavelength [nm]")
    ax.set_ylabel("CE [dB]")
    ax.set_title(f"CE vs signal wavelength for {wl_sep:.2f} nm separation")
    ax.legend()
    # fig.savefig(f"{fig_path}CE_vs_sigwl_{wl_sep:.2f}nm_sep_grouped.pdf", bbox_inches="tight")
