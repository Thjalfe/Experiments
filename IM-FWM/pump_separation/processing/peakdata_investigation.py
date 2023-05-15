import os
import sys
import re

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import load_pump_files
from pump_separation.funcs.processing import (
    sort_multiple_datasets,
    multi_pumpsep_opt_ce,
)
from pump_separation.funcs.plotting import plot_top_n_datasets
import matplotlib.pyplot as plt

# plt.style.use("custom")
plt.ion()

data_folder = "../data/pulse/3us_50duty/manual_specs/"
# data_folder = "../data/pulse/3us_25duty/sortby_redshift/"
# data_folder = "../data/CW/"
data_folder = "../data/CW/2.3W/"
# data_folder = "../data/CW/2.15W/"
# data_folder = "../data/CW/manual/"
file_type = "pkl"
data_folders = ["../data/CW/2.3W/", "../data/CW/2.15W/"]
powers = []

for folder in data_folders:
    match = re.search(r'\d+(\.\d+)?', folder)
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
# Plot the top n datasets
n = 1  # Top n datasets
idx = 6
plot_top_n_datasets(
    red_sorted_peak_data[0][unique_pairs[idx]], raw_data[0][idx], n, unique_pairs[idx]
)
# save_plot(
#     f"./figs/pulsed/CE_top_{n}_datasets_{unique_pairs[idx][0]}_{unique_pairs[idx][1]}_weird_peak"
# )
# |%%--%%| <lGX6ibyli3|N6SM5b714x>
multi_ce_sorted = multi_pumpsep_opt_ce(red_sorted_peak_data, unique_pairs)
idx = 1
fig, ax = plt.subplots(2, 1)
ax[0].plot(
    multi_ce_sorted[idx][0, :],
    np.array(multi_ce_sorted[idx][1, :]),
    marker="o",
    linestyle="-",
)
ax[0].set_ylabel("CE [dB]")
ax[0].grid(True)
ax[1].plot(
    multi_ce_sorted[idx][0, :], multi_ce_sorted[idx][2, :], marker="o", linestyle="-"
)
ax[1].set_xlabel("Pump separation [nm]")
ax[1].set_ylabel("Signal location [nm]")
ax[1].grid(True)

fig, ax = plt.subplots()
for i in range(len(multi_ce_sorted)):
    ax.plot(
        multi_ce_sorted[i][0, :],
        np.array(multi_ce_sorted[i][1, :]),
        marker="o",
        linestyle="-",
        label=f"{powers[i]} W",
    )
ax.set_xlabel("Pump separation [nm]")
ax.set_ylabel("CE [dB]")
ax.legend()
ax.grid(True)

# |%%--%%| <N6SM5b714x|MqLXTtN0Vx>
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
# save_plot(f"./figs/char_setup/{file_str}")
# |%%--%%| <3jwmKKXNgQ|AlpBLQmXLT>
