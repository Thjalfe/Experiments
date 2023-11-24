import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
from pump_separation.funcs.utils import (
    process_single_dataset,
    sort_by_pump_nm_difference,
    extract_pump_wls,
    get_subdirectories,
)  # noqa: E402

plt.style.use("custom")
plt.ion()

# l_band_ce = np.loadtxt('./ce_vs_pumpsep_50duty_Lband_only.csv')
fig_path = "../figs/C_plus_L_band/mean_p_wl_1570/"
save_figs = False
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

upper_folder = "../data/C_plus_L_band/mean_p_wl_1570/"
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


def extract_sig_wl_and_ce_multiple_spectra(
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
    sig_wl_tmp, ce_tmp, idler_wl_tmp = extract_sig_wl_and_ce_multiple_spectra(
        raw_data_specific_pump_sep, pumps, np.shape(raw_data_specific_pump_sep)[0]
    )
    sig_wl.append(sig_wl_tmp)
    ce.append(ce_tmp)
    idler_wl.append(idler_wl_tmp)
pump_seps = [np.max(pump) - np.min(pump) for pump in pump_lst]
# |%%--%%| <CIbwDB3Rgu|bZb4R9CCs4>
# test section for setting up lab scripts
# first create the datastructure that I expect to use in lab
raw_data = {}
iter = 0
for folder in data_folders:
    all_files = sort_file_names_by_filenumber(folder)
    pumps = extract_pump_wls(folder)
    dataset = np.array([np.loadtxt(f, delimiter=",") for f in all_files])
    pump_lst.append(pumps)
    if iter not in raw_data:
        raw_data[iter] = {}
    raw_data[iter]["spectra"] = dataset
    raw_data[iter]["pump_wls"] = pumps
    iter += 1

# |%%--%%| <WKcyLA34wv|CIbwDB3Rgu>
# Section for the changing sig wavelength

fig, ax = plt.subplots()
for i, ce_tmp in enumerate(ce):
    ax.plot(
        sig_wl[i],
        -ce_tmp,
        "-o",
        label=f"{np.abs(np.min(pump_lst[i]) - np.max(pump_lst[i]))}",
    )
ax.legend(title="Pump sep (nm)")
ax.set_xlabel("Signal Wavelength (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
ax.set_title(r"50 \% duty cycle, 100 kHz rep rate, maxed out pump power")
if save_figs:
    plt.savefig(
        f"{fig_path}/CE_vs_sig_wl_diff_pump_seps.pdf",
        bbox_inches="tight",
    )

# |%%--%%| <bZb4R9CCs4|ixhrTmMszv>
# Maximum conversion efficiency vs pump wavelength difference
max_ce_arr = np.zeros(len(ce))
for i, ce_tmp in enumerate(ce):
    max_ce_arr[i] = np.max(-ce_tmp)

fig, ax = plt.subplots()
ax.plot(
    pump_seps,
    max_ce_arr,
    "-o",
)
# ax.plot(

ax.set_xlabel("Pump Wavelength Difference (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
ax.set_title(r"50 \% duty cycle, 100 kHz rep rate, maxed out pump power")
if save_figs:
    plt.savefig(
        f"{fig_path}/max_CE_vs_pump_sep.pdf",
        bbox_inches="tight",
    )
# |%%--%%| <ixhrTmMszv|prIElLNxjr>

sig_wl_at_max = np.zeros(len(ce))
for i, ce_tmp in enumerate(ce):
    max_loc = np.argmax(-ce_tmp)
    sig_wl_at_max[i] = sig_wl[i][max_loc]
y = sig_wl_at_max
fit = np.polyfit(pump_seps, y, 1)
fit_fn = np.poly1d(fit)
fig, ax = plt.subplots()
ax.plot(pump_seps, y, "o", pump_seps, fit_fn(pump_seps), "k")
ax.set_xlabel("Pump Wavelength Difference (nm)")
ax.set_ylabel("Signal Wavelength at Max CE (nm)")
if save_figs:
    plt.savefig(
        f"{fig_path}/sig_wl_at_max_vs_pump_sep.pdf",
        bbox_inches="tight",
    )

np.savetxt("fit_C_plus_L_band_phase_matching_pumpwl_mean=1570.csv", fit, delimiter=",")


# |%%--%%| <prIElLNxjr|5Q2vPtv8ep>
def nm_diff_at_sig_wl(p1_wl, p2_wl, s_wl=970):
    i_wl = 1 / (1 / p1_wl - 1 / p2_wl + 1 / s_wl)
    return i_wl - s_wl


import os
import sys
import matplotlib.pyplot as plt
import numpy as np

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
from pump_separation.funcs.utils import (
    process_single_dataset,
    sort_by_pump_nm_difference,
    extract_pump_wls,
    get_subdirectories,
)  # noqa: E402

plt.style.use("custom")
plt.ion()

# l_band_ce = np.loadtxt('./ce_vs_pumpsep_50duty_Lband_only.csv')
fig_path = "../figs/C_plus_L_band/mean_p_wl_1570/"
save_figs = False
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

upper_folder = "../data/C_plus_L_band/mean_p_wl_1570/"
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
# Section for the changing sig wavelength

fig, ax = plt.subplots()
for i, ce_tmp in enumerate(ce):
    ax.plot(
        sig_wl[i],
        -ce_tmp,
        "-o",
        label=f"{np.abs(np.min(pump_lst[i]) - np.max(pump_lst[i]))}",
    )
ax.legend(title="Pump sep (nm)")
ax.set_xlabel("Signal Wavelength (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
ax.set_title(r"50 \% duty cycle, 100 kHz rep rate, maxed out pump power")
if save_figs:
    plt.savefig(
        f"{fig_path}/CE_vs_sig_wl_diff_pump_seps.pdf",
        bbox_inches="tight",
    )

# |%%--%%| <t2wRldNr8Z|ixhrTmMszv>
# Maximum conversion efficiency vs pump wavelength difference
max_ce_arr = np.zeros(len(ce))
for i, ce_tmp in enumerate(ce):
    max_ce_arr[i] = np.max(-ce_tmp)

fig, ax = plt.subplots()
ax.plot(
    pump_seps,
    max_ce_arr,
    "-o",
)
# ax.plot(

ax.set_xlabel("Pump Wavelength Difference (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
ax.set_title(r"50 \% duty cycle, 100 kHz rep rate, maxed out pump power")
if save_figs:
    plt.savefig(
        f"{fig_path}/max_CE_vs_pump_sep.pdf",
        bbox_inches="tight",
    )
# |%%--%%| <ixhrTmMszv|prIElLNxjr>

sig_wl_at_max = np.zeros(len(ce))
for i, ce_tmp in enumerate(ce):
    max_loc = np.argmax(-ce_tmp)
    sig_wl_at_max[i] = sig_wl[i][max_loc]
y = sig_wl_at_max
fit = np.polyfit(pump_seps, y, 1)
fit_fn = np.poly1d(fit)
fig, ax = plt.subplots()
ax.plot(pump_seps, y, "o", pump_seps, fit_fn(pump_seps), "k")
ax.set_xlabel("Pump Wavelength Difference (nm)")
ax.set_ylabel("Signal Wavelength at Max CE (nm)")
if save_figs:
    plt.savefig(
        f"{fig_path}/sig_wl_at_max_vs_pump_sep.pdf",
        bbox_inches="tight",
    )


# |%%--%%| <prIElLNxjr|5Q2vPtv8ep>
def nm_diff_at_sig_wl(p1_wl, p2_wl, s_wl=970):
    i_wl = 1 / (1 / p1_wl - 1 / p2_wl + 1 / s_wl)
    return i_wl - s_wl
