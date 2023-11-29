import pickle
import os
import re
import glob
from collections import OrderedDict

directory = "../data/sweep_multiple_separations_w_polopt/pol_opt_auto/tisa_sweep_around_opt/mean_p_wl=1590.0"
files = glob.glob(f"{os.path.join(directory, '*.pkl')}")
merged_data = {}
for filename in files:
    if filename.endswith(".pkl"):
        # Extract the numbers from the filename using regex
        match = re.search(r"pump_wl=\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)", filename)
        if match:
            key = (float(match.group(1)), float(match.group(2)))

            # Open the file and load its content
            with open(filename, "rb") as file:
                data = pickle.load(file)

            # Add the content to the dictionary
            merged_data[key] = data


sorted_keys = sorted(merged_data.keys(), key=lambda x: x[0])
sorted_data = OrderedDict((k, merged_data[k]) for k in sorted_keys)
with open(f'{os.path.join(directory, "merged_data.pkl")}', "wb") as file:
    pickle.dump(sorted_data, file)
