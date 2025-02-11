from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from funcs.load_data import get_data_filenames, load_data_as_dict
from funcs.processing import (
    calc_multiple_ces_from_peak_values,
    find_multiple_idler_locs,
    mean_std_data,
)
from funcs.utils import (
    pump_wls_to_thz_sep,
    rolling_average,
    apply_butter_lowpass_filter,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

plt.style.use("custom")
figsize = (11, 7)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = r"\usepackage{upgreek}\usepackage{amsmath}"
save_figs = True
# plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 38
plt.rcParams["ytick.labelsize"] = 38
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"

data_dir = "../../data/"
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
ref_file_names, data_file_names = get_data_filenames(data_dir)
ref_data, duty_cycles = load_data_as_dict(ref_file_names)
data, duty_cycles = load_data_as_dict(data_file_names)
pump_wls_tuple = list(data.keys())
thz_sep = np.array(
    [pump_wls_to_thz_sep(pump_wls_tuple[i]) for i in range(len(pump_wls_tuple))]
)
pump_wls_arr = np.array(pump_wls_tuple)
ref_data_mean_std = mean_std_data(ref_data)
data_mean_std = mean_std_data(data)
idler_locs, idler_wls, sig_wls = find_multiple_idler_locs(data_mean_std)
sig_wl = float(np.mean(sig_wls))
ces, stds = calc_multiple_ces_from_peak_values(
    data_mean_std, ref_data_mean_std, idler_locs, duty_cycles
)
ces = ces * 100
stds = stds * 100


# Now applying filtering/windows on the raw data, more correct processing
ignore_idxs = [5]
num_avg = len(data[pump_wls_tuple[0]]) - 1
data_filtered = np.zeros(
    (
        len(pump_wls_tuple) - len(ignore_idxs),
        num_avg,
        np.shape(data[pump_wls_tuple[-1]])[1],
    )
)
idler_wls_filtered = np.zeros(len(pump_wls_tuple) - len(ignore_idxs))
window_size = 4
min_val = 0.1
count = -1
# Filtering away bad data points and applying a rolling average
for i, pump_wl in enumerate(pump_wls_tuple):
    count += 1
    if i in ignore_idxs:
        count -= 1
        continue
    if np.isnan(data[pump_wl][0, 0]):
        data_tmp = data[pump_wl][1:, 1:]
    else:
        data_tmp = data[pump_wl][1:, :]
    data_tmp[data_tmp < min_val] = min_val
    for j in range(num_avg):
        data_filtered[count, j, :] = rolling_average(data_tmp[j, :], window_size)
    idler_wls_filtered[count] = idler_wls[i]
# Applying a low pass filter to the data
wl_ax = data[pump_wls_tuple[-1]][0, :]
fs = len(wl_ax) / (wl_ax[-1] - wl_ax[0])

data_filtered = apply_butter_lowpass_filter(data_filtered, 10, fs, order=4, axis=2)
idxs_around_idler = 30
idler_peak_wls = np.zeros((len(data_filtered), 2 * idxs_around_idler + 1))
idler_peaks = np.zeros((len(data_filtered), 5, 2 * idxs_around_idler + 1))
# Removing idler peaks from the data and storing them separately
for i in range(len(data_filtered)):
    idler_wl = idler_wls_filtered[i]
    wl_idx = np.nanargmin(np.abs(wl_ax - idler_wl))
    idler_peak_wls[i, :] = wl_ax[
        wl_idx - idxs_around_idler : wl_idx + idxs_around_idler + 1
    ]
    idler_peaks[i, :, :] = data_filtered[
        i, :, wl_idx - idxs_around_idler : wl_idx + idxs_around_idler + 1
    ]
    if i == 1:
        data_filtered[i, :, 650:665] = (
            np.nan
        )  # There was some artifact from the CCTV in this measurement
    # The two following datasets have extra peaks right next to the idler
    if i == 0:
        idler_peaks[i, :, :7] = np.nan
    if i == 1:
        idler_peaks[i, :, :8] = np.nan
    if i == 2:
        idler_peaks[i, :, -19:] = np.nan
        idler_peaks[i, :, :3] = np.nan
    if i == len(data_filtered) - 1:
        idler_peaks[i, :, -18:] = np.nan
    if i == len(data_filtered) - 2:
        idler_peaks[i, :, :8] = np.nan
    data_filtered[i, :, wl_idx - idxs_around_idler : wl_idx + idxs_around_idler + 1] = (
        np.nan
    )
# mean and std of new data filtered
data_filtered_mean = np.nanmean(data_filtered, axis=(0, 1))
data_filtered_std = np.nanstd(data_filtered, axis=(0, 1))
idler_peaks_mean = np.nanmean(idler_peaks, axis=1)
idler_peaks_std = np.nanstd(idler_peaks, axis=1)
max_val_log = 10 * np.log10(np.max(data_filtered_mean))
fig, ax = plt.subplots()
ax = cast(Axes, ax)
fig = cast(Figure, fig)
ax.plot(wl_ax, 10 * np.log10(data_filtered_mean) - max_val_log, color="k", zorder=11)
ax.fill_between(
    wl_ax,
    10 * np.log10(data_filtered_mean - data_filtered_std) - max_val_log,
    10 * np.log10(data_filtered_mean + data_filtered_std) - max_val_log,
    color="grey",
    alpha=1,
    zorder=10,
)
col_count = -1
for i, idler_peak in enumerate(idler_peaks_mean):

    col_count += 1
    if col_count == len(idler_peaks_mean) - 1:
        col_count += 1  # last color does not work well with this plot
    ax.plot(
        idler_peak_wls[i],
        10 * np.log10(idler_peak) - max_val_log,
        color=colors[col_count],
    )
    ax.fill_between(
        idler_peak_wls[i],
        10 * np.log10(idler_peak - idler_peaks_std[i]) - max_val_log,
        10 * np.log10(idler_peak + idler_peaks_std[i]) - max_val_log,
        color=colors[col_count],
        alpha=0.5,
    )
ax.set_ylim(-40, 1)
# fig.subplots_adjust(left=0.2)
ax.set_xlim(np.min(idler_wls) - 1, np.max(idler_wls) + 1)
ax.set_xlim(963.84, 981.43)  # Limits taken ce_vs_detuning_w-classical_lambda_x_ax.pdf
ax.set_xlabel("Wavelength [nm]")
# ax.set_ylabel(r"10$\cdot$log$_{10}$(Counts)")
ax.set_ylabel("Rel. flux [dB]")
ax.yaxis.set_label_coords(-0.118, 0.5)
# fig.tight_layout()

if save_figs:
    fig.savefig(f"{paper_dir}/filtered_spectra_log.pdf", bbox_inches="tight")
# |%%--%%| <gDzQ5NGXwo|zP0Jz79S0T>


pump_wl_ref = pump_wls_tuple[7]
wl_ref_data = ref_data_mean_std[pump_wl_ref]["wl"]
ref_data = ref_data_mean_std[pump_wl_ref]["mean"]
ref_data[ref_data < 0.1] = 0.1
fs = len(ref_data) / (wl_ref_data[-1] - wl_ref_data[1])
ref_data = rolling_average(ref_data, 4)
ref_data = apply_butter_lowpass_filter(ref_data, 10, fs, order=4)

ignore_idxs = [5, 6]
fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
for i, pump_wl in enumerate(pump_wls_tuple):
    if i in ignore_idxs:
        continue
    wl_tmp = data_mean_std[pump_wl]["wl"]
    spec_tmp = data_mean_std[pump_wl]["mean"]
    min_val = 0.1
    spec_tmp[spec_tmp < min_val] = min_val
    spec_tmp = rolling_average(spec_tmp, 6)
    fs = len(wl_tmp) / (wl_tmp[-1] - wl_tmp[1])
    spec_tmp = apply_butter_lowpass_filter(spec_tmp, 10, fs, order=4)
    ax.plot(wl_tmp, 10 * np.log10(spec_tmp))
    wl_idx = np.nanargmin(np.abs(wl_tmp - idler_wls[i]))
    wl = wl_tmp[wl_idx]
    y_val = 10 * np.log10(spec_tmp[wl_idx])

    # Add an arrow annotation at the idler wavelength position
    ax.annotate(
        "",
        xy=(wl, y_val + 10 * 0.2),
        xytext=(wl, y_val + 10 * 1),  # Adjust y positions for arrow and text
        arrowprops=dict(facecolor="black", shrink=0.05),
    )


pump_wl_ref = pump_wls_tuple[7]
wl_ref_data = ref_data_mean_std[pump_wl_ref]["wl"]
ref_data = ref_data_mean_std[pump_wl_ref]["mean"]
ref_data[ref_data < 0.1] = 0.1
fs = len(ref_data) / (wl_ref_data[-1] - wl_ref_data[1])
ref_data = rolling_average(ref_data, 4)
ref_data = apply_butter_lowpass_filter(ref_data, 10, fs, order=4)
ax.plot(
    wl_ref_data,
    10 * np.log10(ref_data),
    color="k",
    label="Reference",
)
ax.set_ylim(1, 50)
ax.set_xlim(np.min(idler_wls) - 1, np.max(idler_wls) + 1)
ax.set_xlabel("Wavelength [nm]")
# ax.set_ylabel(r"10$\cdot$log$_{10}$[Counts]")
ax.set_ylabel(r"10$\cdot$log$_{10}$[Counts]")
if save_figs:
    fig.savefig(f"{paper_dir}/raw_spectra_log.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
for i, pump_wl in enumerate(pump_wls_tuple):
    if i in ignore_idxs:
        continue
    ax.plot(data_mean_std[pump_wl]["wl"], data_mean_std[pump_wl]["mean"])
    wl_idx = np.nanargmin(np.abs(data_mean_std[pump_wl]["wl"] - idler_wls[i]))
    wl = data_mean_std[pump_wl]["wl"][wl_idx]
    y_val = data_mean_std[pump_wl]["mean"][wl_idx]

    # Add an arrow annotation at the idler wavelength position
    ax.annotate(
        "",
        xy=(wl, y_val + 500),
        xytext=(wl, y_val + 10000),  # Adjust y positions for arrow and text
        arrowprops=dict(facecolor="black", shrink=100),
    )


pump_wl_ref = pump_wls_tuple[7]
ax.plot(
    ref_data_mean_std[pump_wl_ref]["wl"],
    ref_data_mean_std[pump_wl_ref]["mean"],
    color="k",
    label="Reference",
)
# ax.set_ylim([1, 5])
ax.set_xlim(np.min(idler_wls) - 1, np.max(idler_wls) + 1)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Counts")
if save_figs:
    fig.savefig(f"{paper_dir}/raw_spectra.pdf", bbox_inches="tight")
plt.show()
