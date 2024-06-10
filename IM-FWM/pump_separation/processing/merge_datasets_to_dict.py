# %%
import pickle
import os
import re
import glob
from collections import OrderedDict

# directory = "../data/modelocked_1571_pump/cw_pumps_to_find_phasematch/"
directory = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\4MSI\tisa_sweep_to_find_opt\sig_idler_modes=01_equal_input_pump_p"
directory = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\4MSI\tisa_sweep_to_find_opt\pump_modes=21_skewed_input_pump_p"
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
