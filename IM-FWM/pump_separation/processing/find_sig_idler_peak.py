import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.ion()

# Load data
with open(
    "../data/modelocked_1571_pump/cw_pumps_to_find_phasematch/merged_data.pkl", "rb"
) as f:
    data = pickle.load(f)

p1_wls = list(data.keys())
p1_wl = p1_wls[0]
# p2_wl = p2_wls[0]
# sub_data = data[p1_wl][p2_wl]
# wl_arr = sub_data[0, :]
# power_arr = sub_data[1, :]


# |%%--%%| <Dw82hl7Vtw|DS7yry3rgt>
import os

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
import sys

sys.path.append(os.path.normpath(project_root_dir))
from pump_separation.funcs.utils import process_single_dataset  # noqa: E402, E501

processed_data = {}
for p1_wl in p1_wls:
    processed_data[p1_wl] = {}
    p2_wls = list(data[p1_wl].keys())
    for p2_wl in p2_wls:
        sub_data = data[p1_wl][p2_wl]
        processed_data[p1_wl][p2_wl] = process_single_dataset(
            sub_data.T, 0.1, -70, tuple((p1_wl, p2_wl)), 0.1, -25
        )

ce_vs_p2wl_for_all_p1wl = {}
for p1_wl in p1_wls:
    ce_vs_p2wl_for_all_p1wl[p1_wl] = []
    p2_wls = list(data[p1_wl].keys())
    for p2_wl in p2_wls:
        processed_data_tmp = processed_data[p1_wl][p2_wl]
        if type(processed_data_tmp) == str:
            ce_vs_p2wl_for_all_p1wl[p1_wl].append(np.nan)
        else:
            ce_vs_p2wl_for_all_p1wl[p1_wl].append(processed_data_tmp["differences"][0])
fig, ax = plt.subplots()
for p1_wl in p1_wls:
    ax.plot(p2_wls, ce_vs_p2wl_for_all_p1wl[p1_wl], label=f"{p1_wl} nm")
ax.legend()
