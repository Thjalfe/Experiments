# This script merges two datasets and processes the merged data
from process_datasets import add_data_process_dict_to_data_dict

import pickle
import numpy as np
import os

old_data = "./sweep_both_pumps_auto_pol_opt_780-fiber-out/p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40/data.pkl"
new_data = "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/p1-wl=1591-1596.25nm_p2-wl=1608.60-1570.69_merged_v2/data.pkl"
with open(old_data, "rb") as f:
    old = pickle.load(f)
with open(new_data, "rb") as f:
    new = pickle.load(f)

merged_dict = {}
new_dict_only = {}
for key in new.keys():
    merged_dict[key] = new[key]
    new_dict_only[key] = new[key]
    if isinstance(key, (np.floating, np.integer)):
        new_dict_only[key] = new[key]
    else:
        merged_dict[key] = old[key]
old_keys_to_igore_or_overwrite = [1592.0, 1592.8, 1595, 1596]
# |%%--%%| <8x12JwjHm7|Cye7EX5xWe>
for key in old.keys():
    if key in old_keys_to_igore_or_overwrite:
        continue
    if key not in merged_dict.keys():
        if isinstance(key, (np.floating, np.integer)):
            merged_dict[key] = old[key]

for key in merged_dict.keys():
    if isinstance(key, (np.floating, np.integer)):
        if key in new_dict_only.keys():
            merged_dict[key]["duty_cycle"] = new_dict_only["duty_cycle"]
        elif key in old.keys():
            merged_dict[key]["duty_cycle"] = old["duty_cycle"]
float_keys = {
    k: v
    for k, v in merged_dict.items()
    if isinstance(k, (np.floating, np.integer))
    and isinstance(v, dict)
    and "p2_max_ce" in v
}
non_float_keys = {k: v for k, v in merged_dict.items() if not isinstance(k, float)}

# Sort the float keys in ascending order and reinsert the non-float keys
sorted_float_keys = sorted(float_keys)
sorted_merged_dict = {k: float_keys[k] for k in sorted_float_keys}
sorted_merged_dict.update(non_float_keys)


data_dict = add_data_process_dict_to_data_dict(sorted_merged_dict)
dir_name = f"./sweep_both_pumps_w_processed_merged_proper_data/p1-wl={sorted_float_keys[0]}-{sorted_float_keys[-1]}nm_p2-wl={sorted_merged_dict[sorted_float_keys[0]]['p2_max_ce']:.2f}-{sorted_merged_dict[sorted_float_keys[-1]]['p2_max_ce']:.2f}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(f"{dir_name}/data_processed.pkl", "wb") as f:
    pickle.dump(data_dict, f)
