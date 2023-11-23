import os
import glob
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
from pump_separation.funcs.utils import process_single_dataset, sort_by_pump_nm_difference, extract_pump_wls, get_subdirectories  # noqa: E402

plt.style.use("custom")
plt.ion()




fig_path = "../figs/C_plus_L_band/idler_stability_meas/"
save_figs = False
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

upper_folder = "../data/C_plus_L_band/idler_stability_meas/"
# upper_folder = "../data/C_plus_L_band/idler_stability_meas/with_box/"
data_folders = get_subdirectories(upper_folder)
data_folders = [os.path.join(upper_folder, folder) for folder in data_folders]
for i, data_folder in enumerate(data_folders):
    if not data_folder.endswith("/"):
        data_folder += "/"
        data_folders[i] = data_folder
    if len(extract_pump_wls(data_folder)) == 0:
        # remove folders that don't have numbers in them
        data_folders.remove(data_folder)
data_folders = sort_by_pump_nm_difference(data_folders)
file_type = "csv"


def sort_file_names_by_filenumber(data_folder):
    all_files = glob.glob(data_folder + "*." + file_type)
    all_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return all_files


def extract_sig_wl_and_ce(
    data,
    pump_lst,
    num_files,
    prominence=0.1,
    height=-75,
    tolerance_nm=0.25,
    max_peak_min_height=-35,
):
    sig_wl = np.zeros(num_files)
    ce = np.zeros(num_files)
    idler_wl = np.zeros(num_files)
    for j in range(num_files):
        processed = process_single_dataset(
            data[j, :, :],
            prominence,
            height,
            (pump_lst[0], pump_lst[1]),
            tolerance_nm,
            max_peak_min_height,
        )
        sig_loc = np.argmax(processed["differences"])
        sig_wl[j] = processed["peak_positions"][sig_loc]
        idler_wl[j] = processed["peak_positions"][sig_loc + 1]
        ce[j] = processed["differences"][-1]
    return sig_wl, ce, idler_wl


raw_data = []
sig_wl = []
ce = []
idler_wl = []
noise_floor = []
pump_lst = []
for folder in data_folders:
    all_files = sort_file_names_by_filenumber(folder)
    pumps = extract_pump_wls(folder)
    dataset = np.array([np.loadtxt(f, delimiter=",") for f in all_files])
    raw_data.append(dataset)
    pump_lst.append(pumps)

for idx, raw_data_specific_pump_sep in enumerate(raw_data):
    pumps = pump_lst[idx]
    sig_wl_tmp, ce_tmp, idler_wl_tmp = extract_sig_wl_and_ce(
        raw_data_specific_pump_sep, pumps, np.shape(raw_data_specific_pump_sep)[0]
    )
    sig_wl.append(sig_wl_tmp)
    ce.append(ce_tmp)
    idler_wl.append(idler_wl_tmp)
pump_seps = [np.max(pump) - np.min(pump) for pump in pump_lst]
# |%%--%%| <WKcyLA34wv|sIg8wX78ou>
# Find indices close to the signal and idler wavelengths
sig_wl_idx = []
idler_wl_idx = []
for i, dataset in enumerate(raw_data):
    mean_sig_wl = np.mean(sig_wl[i])
    mean_idler_wl = np.mean(idler_wl[i])
    # indices within 0.5 nm of the mean signal wavelength
    sig_wl_idx.append(
        np.where(
            np.logical_and(
                dataset[0, :, 0] > mean_sig_wl - 0.25,
                dataset[0, :, 0] < mean_sig_wl + 0.25,
            )
        )[0]
    )
    # indices within 0.5 nm of the mean idler wavelength
    idler_wl_idx.append(
        np.where(
            np.logical_and(
                dataset[0, :, 0] > mean_idler_wl - 0.25,
                dataset[0, :, 0] < mean_idler_wl + 0.25,
            )
        )[0]
    )
# |%%--%%| <sIg8wX78ou|uFz3j8EYCG>
for iter, dataset in enumerate(raw_data):
    pumps = pump_lst[iter]
    fig, ax = plt.subplots()
    sig_wl_idx_tmp = sig_wl_idx[iter]
    sig_start, sig_end = sig_wl_idx_tmp[0], sig_wl_idx_tmp[-1]
    idler_wl_idx_tmp = idler_wl_idx[iter]
    idler_start, idler_end = idler_wl_idx_tmp[0], idler_wl_idx_tmp[-1]
    axins1 = ax.inset_axes([0.3, 0.6, 0.3, 0.3])
    axins2 = ax.inset_axes([0.7, 0.6, 0.3, 0.3])
    for i in range(np.shape(dataset)[0]):
        ax.plot(dataset[i, :, 0], dataset[i, :, 1])
        axins2.plot(
            dataset[i, idler_start:idler_end, 0], dataset[i, idler_start:idler_end, 1]
        )
        axins1.plot(dataset[i, sig_start:sig_end, 0], dataset[i, sig_start:sig_end, 1])
    lower_idler_lim, upper_idler_lim = axins2.get_ylim()
    idler_extinction = upper_idler_lim - lower_idler_lim
    axins2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    axins1.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")
    lower_sig_lim, upper_sig_lim = axins1.get_ylim()
    axins1.set_ylim(upper_sig_lim - idler_extinction, upper_sig_lim)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (dBm)")
    ax.set_title(f"Pump sep {np.max(pumps) - np.min(pumps):.2f} nm")
    if save_figs:
        plt.savefig(
            f"{fig_path}/{np.max(pumps) - np.min(pumps):.2f}nm_pump_sep_rawdata.pdf",
            bbox_inches="tight",
        )
# |%%--%%| <uFz3j8EYCG|BjzwvpxCc4>
# mean and std of the data
mean_data = []
std_data = []
for dataset in raw_data:
    mean_data.append(np.mean(dataset, axis=0))
    std_data.append(np.std(dataset, axis=0))
for i, mean in enumerate(mean_data):
    pumps = pump_lst[i]
    sig_wl_idx_tmp = sig_wl_idx[i]
    sig_start, sig_end = sig_wl_idx_tmp[0], sig_wl_idx_tmp[-1]
    idler_wl_idx_tmp = idler_wl_idx[i]
    idler_start, idler_end = idler_wl_idx_tmp[0], idler_wl_idx_tmp[-1]
    fig, ax = plt.subplots()
    ax.plot(mean[:, 0], mean[:, 1], label=f"{pump_lst[i][0]} - {pump_lst[i][1]}")
    ax.fill_between(
        mean[:, 0],
        mean[:, 1] - std_data[i][:, 1],
        mean[:, 1] + std_data[i][:, 1],
        alpha=0.3,
    )
    axins2 = ax.inset_axes([0.7, 0.6, 0.3, 0.3])
    axins2.plot(mean[idler_start:idler_end, 0], mean[idler_start:idler_end, 1])
    axins2.fill_between(
        mean[idler_start:idler_end, 0],
        mean[idler_start:idler_end, 1] - std_data[i][idler_start:idler_end, 1],
        mean[idler_start:idler_end, 1] + std_data[i][idler_start:idler_end, 1],
        alpha=0.3,
    )
    axins2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    lower_idler_lim, upper_idler_lim = axins2.get_ylim()
    idler_extinction = upper_idler_lim - lower_idler_lim
    axins1 = ax.inset_axes([0.3, 0.6, 0.3, 0.3])
    axins1.plot(mean[sig_start:sig_end, 0], mean[sig_start:sig_end, 1])
    axins1.fill_between(
        mean[sig_start:sig_end, 0],
        mean[sig_start:sig_end, 1] - std_data[i][sig_start:sig_end, 1],
        mean[sig_start:sig_end, 1] + std_data[i][sig_start:sig_end, 1],
        alpha=0.3,
    )
    axins1.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")
    lower_sig_lim, upper_sig_lim = axins1.get_ylim()
    axins1.set_ylim(upper_sig_lim - idler_extinction, upper_sig_lim)
    # ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (dBm)")
    ax.set_title(
        f"Mean and Std of Pump Wavelengths: {pump_lst[i][0]} nm, {pump_lst[i][1]} nm"
    )
    if save_figs:
        plt.savefig(
            f"{fig_path}/{np.max(pumps) - np.min(pumps):.2f}nm_pump_sep_meandata_w_box.pdf",
            bbox_inches="tight",
        )
