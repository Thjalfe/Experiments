import os
import sys
current_file_path = os.path.abspath('.py')
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, '..', '..')
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import (
    analyze_data,
    load_raw_data,
    get_all_unique_pairs_list,
    load_pump_files,
)
from pump_separation.funcs.plotting import save_plot, plot_top_n_datasets
import matplotlib.pyplot as plt

plt.style.use("custom")
plt.ion()

data_folder = "../data/pulsed/SMF_pickup_hi1060/first_dataset/"
# data_folder = "../data/pulsed/"
# data_folder = "../data/pulsed/MMF_pickup/"
file_type = "pkl"
unique_pairs = get_all_unique_pairs_list(data_folder, file_type=file_type)
# unique_pairs = (1569, 1573)
data, chosen_pairs, other = load_raw_data(
    data_folder, -85, None, unique_pairs, file_type=file_type
)
(
    sorted_peak_data,
    blueshift_sorted_peak_data,
    redshift_sorted_peak_data,
) = analyze_data(data_folder, pump_wl_pairs=unique_pairs, file_type=file_type)
dataset_cur = redshift_sorted_peak_data
# |%%--%%| <bUxbvLaNgk|58VOLX1DCi>
# Plot the top n datasets
n = 4  # Top n datasets
idx = 0
plot_top_n_datasets(
    dataset_cur[unique_pairs[idx]], data[idx], n, unique_pairs[idx]
)
# save_plot(
#     f"./figs/pulsed/CE_top_{n}_datasets_{unique_pairs[idx][0]}_{unique_pairs[idx][1]}_weird_peak"
# )
# |%%--%%| <58VOLX1DCi|79QO9wN6Wo>
best_ce = []
best_ce_loc = []
best_idx = []
for i, key in enumerate(chosen_pairs):
    best_idx.append(list(dataset_cur[key].keys())[0])
    best_ce.append(min(dataset_cur[key][best_idx[i]]["differences"]))
    temp_best_loc = np.argmin(dataset_cur[key][best_idx[i]]["differences"])
    best_ce_loc.append(
        dataset_cur[key][best_idx[i]]["peak_positions"][temp_best_loc]
    )
    # best_ce.append(dataset_cur[key][0])
    # best_ce_loc.append(dataset_cur[key][1])
x = np.arange(1, len(best_ce) + 1)
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, -np.array(best_ce), marker="o", linestyle="-")
ax[0].set_ylabel("CE [dB]")
ax[0].grid(True)
ax[1].plot(x, best_ce_loc, marker="o", linestyle="-")
ax[1].set_xlabel("Pump separation [nm]")
ax[1].set_ylabel("Signal location [nm]")
ax[1].grid(True)
# also save the data that is plotted here as a csv file
import pandas as pd
dataframe = pd.DataFrame({'CE': best_ce, 'CE_loc': best_ce_loc, 'pump_sep': x})
dataframe.to_csv('CE_vs_pump_separation.csv')

# save_plot("./figs/pulsed/CE_vs_pump_separation")
# fig.savefig('CE_vs_pump_separation.png')
# |%%--%%| <79QO9wN6Wo|Ftmiqvm323>
from scipy.signal import find_peaks


char_folder = "./char_setup/"
file_str = "pump_sep_through_setup_"
file_str = "pumps_directly_to_osa"
file_str = "pump_directly_to_osa"
# file_str = 'before_edfa'
pumps, unique_pump_pairs = load_pump_files(char_folder, pump_name=file_str)

fig, ax = plt.subplots()
for i in range(pumps.shape[0]):
    peaks, _ = find_peaks(pumps[i, :, 1], height=-80)
    peaks_max = peaks[np.argsort(pumps[i, peaks, 1])[-2:]]
    ax.plot(
        pumps[i, :, 0],
        pumps[i, :, 1],
        label=f"{unique_pump_pairs[i]} nm, power diff: {pumps[i, peaks_max[0], 1] - pumps[i, peaks_max[1], 1]:.2f} dB",
    )
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Power [dBm]")
ax.grid(True)
# ax.legend(loc="lower right")
# plt.show()
save_plot(f"./figs/char_setup/{file_str}")
