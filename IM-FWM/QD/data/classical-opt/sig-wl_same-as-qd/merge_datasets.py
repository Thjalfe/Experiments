# This script merges datasets from different dirs in a super_dir
import os
import pickle
from glob import glob

import numpy as np


# make a function for different merge cases
def merge_case(case_name: str):
    super_dir = None
    extra_files = []
    extra_files_keys_to_unpack = None
    back_to_equal_p_case = False
    match case_name:
        case "equal_powers_aka-unoptimized-red-side":
            # THERE IS A WHY.TXT FILE IN THIS DIR THAT EXPLAINS WHY THIS DATA EVEN EXISTS
            super_dir = "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/day4_back_to_equal_p1_p2_powers"
            # EXTRA FILES ARE OTHER DATASET WHERE THE PUMP POWERS WERE EQUAL OUT OF THE BOX
            # FOR SOME OF THESE, THIS IS OPTIMAL WHICH IS WHY IT IS USED MULTIPLE DIFFERENT TIMES
            # SEE PLOTS IF IN DOUBT
            extra_files = [
                "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/old_worse_data/p1_wl=1594.8-1594.8_p2_wl=1576.8-1582.2_p1_stepsize=0.50_p2_stepsize=0.10/data.pkl",
                "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/day3/p1_wl=1593.5_p2_wl=1591.5_manual_fine-tuning/data.pkl",
                "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/day3/p1_wl=1593.8_p2_wl=1589.9_manual_fine-tuning/data.pkl",
                "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/day3/p1_wl=1594.0_p2_wl=1588.4_manual_fine-tuning/data.pkl",
            ]
            extra_files_keys_to_unpack = "./sweep_both_pumps_auto_pol_opt_780-fiber-out/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40/data.pkl"
            back_to_equal_p_case = True
        case "optimized-powers":
            super_dir = "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/"
            back_to_equal_p_case = False
            extra_files = []
            extra_files_keys_to_unpack = None
    return super_dir, extra_files, extra_files_keys_to_unpack, back_to_equal_p_case


cases = ["equal_powers_aka-unoptimized-red-side", "optimized-powers"]
case = cases[1]
super_dir, extra_files, extra_files_keys_to_unpack, back_to_equal_p_case = merge_case(
    case
)
all_data = {}
file_names = glob(f"{super_dir}/**/data.pkl", recursive=True)
file_names.extend(extra_files)
for data_file in file_names:
    if "merged" in data_file:
        continue
    if "old_worse_data" in data_file:
        continue
    if not back_to_equal_p_case and "back_to_equal_p" in data_file:
        continue
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    all_data.update(data)
if extra_files_keys_to_unpack is not None:
    with open(extra_files_keys_to_unpack, "rb") as f:
        extra_data = pickle.load(f)
    key_idxs_to_use = np.array(
        [-3]
    )  # hard coded because this is the only key we want to use from this dataset
    extra_data_keys = list(extra_data.keys())
    extra_data_numerical_keys = np.array(
        [float(k) for k in extra_data_keys if isinstance(k, (float, int))]
    )
    extra_data_keys_to_use = np.array(extra_data_numerical_keys)[key_idxs_to_use]
    for key in extra_data_keys_to_use:
        if key in all_data:
            continue
        all_data[key] = extra_data[key]
# |%%--%%| <7dgqUdbAC0|SCaJxNl7Pp>

numeric_keys = {
    k: v
    for k, v in all_data.items()
    if isinstance(k, (float, int, np.floating, np.integer))
    and isinstance(v, dict)
    and "p2_max_ce" in v
}
non_numeric_keys = {
    k: v
    for k, v in all_data.items()
    if not isinstance(k, (float, int, np.floating, np.integer))
}
sorted_numeric_keys = sorted(numeric_keys)
sorted_merged_dict = {k: numeric_keys[k] for k in sorted_numeric_keys}
sorted_merged_dict.update(non_numeric_keys)
dir_name = os.path.join(
    super_dir,
    f"p1-wl={sorted_numeric_keys[0]}-{sorted_numeric_keys[-1]}nm_p2-wl={sorted_merged_dict[sorted_numeric_keys[0]]['p2_max_ce']:.2f}-{sorted_merged_dict[sorted_numeric_keys[-1]]['p2_max_ce']:.2f}_merged_v2",
)
if os.path.exists(dir_name):
    dir_name = f"{dir_name}_1"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open(f"{dir_name}/data.pkl", "wb") as f:
    pickle.dump(sorted_merged_dict, f)
