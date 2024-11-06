import pickle
import os

red_data = "./p1_wl=1593.8-1595.8_p2_wl=1575.6-1591.7_p1_stepsize=0.50_p2_stepsize=0.10/data.pkl"
blue_data = "./p1_wl=1590.8-1592.8_p2_wl=1591.4-1611.0_p1_stepsize=0.50_p2_stepsize=0.10/data.pkl"
with open(red_data, "rb") as f:
    red = pickle.load(f)
with open(blue_data, "rb") as f:
    blue = pickle.load(f)
merged_dict = {**red, **blue}

# Separate float keys and other keys (like 'duty_cycle', etc.)
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
