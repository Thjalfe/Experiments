import pickle
import numpy as np
import os

old_data = "./p1-wl=1590.5-1596.0nm_p2-wl=1610.60-1572.40/data.pkl"
new_data = "./p1_wl=1593.8-1595.8_p2_wl=1573.5-1591.7_p1_stepsize=0.50_p2_stepsize=0.10/data.pkl"
with open(old_data, "rb") as f:
    old = pickle.load(f)
with open(new_data, "rb") as f:
    new = pickle.load(f)
merged_dict = {}
new_dict_only = {}
for key in new.keys():
    if key not in old.keys():
        merged_dict[key] = new[key]
        new_dict_only[key] = new[key]
    elif type(key) is not np.float64:
        new_dict_only[key] = new[key]
    else:
        merged_dict[key] = old[key]
for key in old.keys():
    if key not in merged_dict.keys():
        merged_dict[key] = old[key]

float_keys = {
    k: v
    for k, v in merged_dict.items()
    if isinstance(k, float) and isinstance(v, dict) and "p2_max_ce" in v
}
non_float_keys = {k: v for k, v in merged_dict.items() if not isinstance(k, float)}

# Sort the float keys in ascending order and reinsert the non-float keys
sorted_float_keys = sorted(float_keys)
sorted_merged_dict = {k: float_keys[k] for k in sorted_float_keys}
sorted_merged_dict.update(non_float_keys)

dir_name = f"p1-wl={sorted_float_keys[0]}-{sorted_float_keys[-1]}nm_p2-wl={sorted_merged_dict[sorted_float_keys[0]]['p2_max_ce']:.2f}-{sorted_merged_dict[sorted_float_keys[-1]]['p2_max_ce']:.2f}_2"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(f"{dir_name}/data.pkl", "wb") as f:
    pickle.dump(sorted_merged_dict, f)
