import os
import glob
import sys
import re
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import process_single_dataset

plt.ion()

data_folders = [
    "../data/C_plus_L_band/mean_p_wl_1566.5/1570.00nm_1563.00nm/",
    "../data/C_plus_L_band/mean_p_wl_1566.5/1581.50nm_1551.50nm/",
    "../data/C_plus_L_band/mean_p_wl_1566.5/1591.50nm_1541.50nm/",
    "../data/C_plus_L_band/mean_p_wl_1570/1605.00nm_1535.00nm/",
]


def get_subdirectories(directory):
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]

def sort_by_nm_difference(paths):
    def nm_difference(path):
        # Extract the nm numbers using regular expressions
        numbers = re.findall(r'(\d+.\d+)nm', path)
        # Convert the extracted numbers to floats and calculate their difference
        return abs(float(numbers[-1]) - float(numbers[-2]))

    # Sort the list based on the nm difference
    return sorted(paths, key=nm_difference)


upper_folder = "../data/C_plus_L_band/mean_p_wl_1570/"
data_folders = get_subdirectories(upper_folder)
data_folders = [os.path.join(upper_folder, folder) for folder in data_folders]
for i, data_folder in enumerate(data_folders):
    if not data_folder.endswith("/"):
        data_folder += "/"
        data_folders[i] = data_folder
data_folders = sort_by_nm_difference(data_folders)
file_type = "csv"
sig_wl = []
ce = []
pump_lst = []


def extract_numbers(s):
    numbers = re.findall(r"\d+\.\d+", s)
    numbers = [float(num) for num in numbers]
    numbers = numbers[-2:]
    numbers.sort()
    return numbers


def sorted_file_names(data_folder):
    all_files = glob.glob(data_folder + "*." + file_type)
    all_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return all_files


def extract_sig_wl_and_ce(
    data_folder,
    pump_lst,
    prominence=0.1,
    height=-75,
    tolerance_nm=0.25,
    max_peak_min_height=-35,
):
    all_files = sorted_file_names(data_folder)
    dataset = np.array([np.loadtxt(f, delimiter=",") for f in all_files])
    sig_wl = np.zeros(len(dataset))
    ce = np.zeros(len(dataset))
    for j, data in enumerate(dataset):
        processed = process_single_dataset(
            data,
            prominence,
            height,
            (pump_lst[0], pump_lst[1]),
            tolerance_nm,
            max_peak_min_height,
        )
        sig_loc = np.argmax(processed["differences"])
        sig_wl[j] = processed["peak_positions"][sig_loc]
        ce[j] = processed["differences"][-1]
    return sig_wl, ce

ce_max = []
for folder in data_folders:
    pumps = extract_numbers(folder)
    sig_wl_tmp, ce_tmp = extract_sig_wl_and_ce(folder, pumps)
    sig_wl.append(sig_wl_tmp)
    ce.append(ce_tmp)
    pump_lst.append(pumps)
    ce_max.append(np.max(-ce_tmp))
# |%%--%%| <t2wRldNr8Z|U0tovuxtiW>
sig_wl_at_max = np.zeros(len(ce))
for i, ce_tmp in enumerate(ce):
    max_loc = np.argmax(-ce_tmp)
    sig_wl_at_max[i] = sig_wl[i][max_loc]
x = [np.max(pump) - np.min(pump) for pump in pump_lst]
y = sig_wl_at_max
fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
# |%%--%%| <U0tovuxtiW|wJE5tmQHCh>

fig, ax = plt.subplots()
for i, ce_tmp in enumerate(ce):
    ax.plot(
        sig_wl[i], -ce_tmp, "-o", label=f"{np.max(pump_lst[i]) - np.min(pump_lst[i])}"
    )
ax.legend()
ax.set_xlabel("Signal Wavelength (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
#|%%--%%| <wJE5tmQHCh|IqnW2HuT4Q>
fig, ax = plt.subplots()
ax.plot(x, ce_max, "-o")
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
# |%%--%%| <IqnW2HuT4Q|5Q2vPtv8ep>
def nm_diff_at_sig_wl(p1_wl, p2_wl, s_wl=970):
    i_wl = 1 / (1 / p1_wl - 1 / p2_wl + 1 / s_wl)
    return i_wl - s_wl
