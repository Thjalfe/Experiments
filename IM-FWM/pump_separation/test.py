from util_funcs import *
import glob
from scipy.signal import find_peaks

# from experiment_funcs import new_sig_start


data_folder = "./data/pulsed/SMF_pickup_hi1060_test/"
chosen_pairs = [(1570.0, 1572.0)]
pump_wllow = 1570.0
pump_wlhigh = 1572.0
wl_tot = 0.2
file_type = "pkl"
data = load_raw_data(data_folder, -80, pump_wl_pairs=chosen_pairs)
nan_filter = -85
prominence = 0.1
height = -65
tolerance_nm = 0.25
max_peak_height = -25
# peaks_sorted = analyze_data(data_folder, pump_wl_pairs=[(pumpwl_low, pumpwl_high)])[
#     (pumpwl_low, pumpwl_high)
# ]
for i, dataset in enumerate(data):
    peak_data = process_dataset(
        dataset, prominence, height, chosen_pairs[i], tolerance_nm, max_peak_height
    )
    sorted_peak_data = sort_peak_data(peak_data, "differences")
    all_sorted_peak_data[chosen_pairs[i]] = sorted_peak_data
# data_folder = data_folder[:-1]
# file_list = natsorted(glob.glob(data_folder + f"/*[0-9].{file_type}"))
