import os
from pump_separation.funcs.utils import (
    extract_sig_wl_and_ce_single_spectrum,
)
import pickle
import glob
import matplotlib.pyplot as plt
from curlyBrace import curlyBrace
import numpy as np
from matplotlib.ticker import MaxNLocator
from pump_separation.funcs.process_multiple_spectra_sorted_in_dicts import (
    process_ce_data_for_pump_sweep_simple,
)
import re

plt.style.use("large_fonts")
# plt.rcParams["figure.figsize"] = (16, 11)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()
data_loc_tisa965 = "../data/TMSI_tisa=965.1_pump_L_band_sweep/spectra.pkl"
with open(data_loc_tisa965, "rb") as f:
    data_tisa965 = pickle.load(f)
fig_folder = "../../figs/sweep_multiple_separations_w_polopt/cleo_us_2023/pol_opt_auto"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

save_figs = False
save_spectra = False
merge_data = True
pump2_wls = data_tisa965[1600]["pump2_wls"]
pump1_wls = list(data_tisa965.keys())
(
    exp_params_tisa965,
    ce_dict_tisa965,
    sig_wl_dict_tisa965,
    idler_wl_dict_tisa965,
    ce_dict_processed_tisa965,
    sig_wl_dict_processed_tisa965,
    idler_wl_dict_processed_tisa965,
    ce_dict_std_tisa965,
    ce_dict_best_for_each_ando_sweep_tisa965,
    ce_dict_best_loc_for_each_ando_sweep_tisa965,
) = process_ce_data_for_pump_sweep_simple(data_tisa965, np.mean)
# |%%--%%| <4lwt60hUsn|KwOOxrg3l0>
pump1_wl = pump1_wls[5]
ce_tmp = ce_dict_processed_tisa965[pump1_wl][0.1]
fig, ax = plt.subplots()
ax.plot(pump2_wls, ce_tmp)
ax.set_title(f"CE for pump1_wl={pump1_wl} nm")
plt.show()
# |%%--%%| <KwOOxrg3l0|xC3x1AQcjf>
### Different datatype
from pump_separation.funcs.utils import (
    extract_sig_wl_and_ce_single_spectrum,
)

data_dir = "../data/TMSI_tisa=965.1_pump_L_band_sweep/manual_precise_sweep/0.05"
# data_dir = "../data/TMSI_tisa=965.1_pump_L_band_sweep/manual_precise_sweep/0.1"
dc = float(data_dir.split("/")[-1])
dc_offset = 10 * np.log10(dc)
file_names = glob.glob(os.path.join(data_dir, "*.pkl"))
if dc == 0.05:
    file_name_idxs_ignore = [2, 3, 9]
    # file_name_idxs_ignore = [3, 8, 9]
else:
    file_name_idxs_ignore = [3]

pump_wl_arr = np.zeros((len(file_names) - len(file_name_idxs_ignore), 2))
iter = 0
ce = [[] for _ in range(len(file_names) - len(file_name_idxs_ignore))]
sig_wls = [[] for _ in range(len(file_names) - len(file_name_idxs_ignore))]
idler_wls = [[] for _ in range(len(file_names) - len(file_name_idxs_ignore))]
mean_ce = np.zeros((len(file_names) - len(file_name_idxs_ignore)))
for cur_filename in file_names:
    if file_names.index(cur_filename) in file_name_idxs_ignore:
        continue
    with open(cur_filename, "rb") as f:
        cur_data = pickle.load(f)
    # Use regex to extract the numbers
    match = re.search(r"spec_(\d+\.\d+)_(\d+\.\d+)", cur_filename)

    pump1_wl = float(match.group(1)) if match else None
    pump2_wl = float(match.group(2)) if match else None
    pump_wl_lst = [pump1_wl, pump2_wl]
    pump_wl_arr[iter, :] = pump_wl_lst
    if len(np.shape(cur_data)) == 2:
        cur_data = np.expand_dims(cur_data, axis=0)
    num_reps = np.shape(cur_data)[0]
    for i in range(num_reps):
        tmp_data = cur_data[i].T
        sig_wl, ce_tmp, idler_wl = extract_sig_wl_and_ce_single_spectrum(
            tmp_data, pump_wl_lst, tolerance_nm=0.5
        )
        ce[iter].append(-ce_tmp - dc_offset)
        sig_wls[iter].append(sig_wl)
        idler_wls[iter].append(idler_wl)
    mean_ce[iter] = np.mean(ce[iter])
    iter += 1
sorted_idxs = np.argsort(pump_wl_arr[:, 0])
pump_wl_arr = pump_wl_arr[sorted_idxs]
ce = [ce[i] for i in sorted_idxs]
sig_wls = [sig_wls[i] for i in sorted_idxs]
idler_wls = [idler_wls[i] for i in sorted_idxs]
mean_ce = mean_ce[sorted_idxs]
pump_wl_seps = np.abs(
    [pump_wl_arr[i, 1] - pump_wl_arr[i, 0] for i in range(len(pump_wl_arr))]
)

fig, ax = plt.subplots()
ax.plot(pump_wl_seps, mean_ce)
# ax2 = ax.twiny()
# ax2.plot(pump_wl_arr[:, 0], mean_ce)
ax.set_xlabel("Pump separation (nm)")
ax.set_ylabel("CE (dB)")
ax.set_title(f"CE for duty cycle={dc}")
