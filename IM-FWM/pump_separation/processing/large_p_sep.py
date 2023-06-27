import os
import pickle
import sys
import re

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import load_pump_files, process_multiple_datasets
from pump_separation.funcs.processing import (
    sort_multiple_datasets,
    multi_pumpsep_opt_ce,
    get_subfolders,
)
import matplotlib.pyplot as plt
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.ion()
plt.rcParams["figure.figsize"] = (20, 11)
file_names = ["pumps_1568.0_1608.0_spectra.pkl", "pumps_1570.5_1605.5_spectra.pkl", "pumps_1573.0_1603.0_spectra.pkl", "pumps_1575.5_1600.5_spectra.pkl"]
filepath = "../data/large_pump_sep_within_L_band/"
data = []
for i, file_name in enumerate(file_names):
    with open(filepath + file_name, "rb") as f:
        data.append(pickle.load(f))

# data = np.loadtxt(filepath, delimiter=",", skiprows=1)
#|%%--%%| <4WpiAjEyX6|fPNSjlkTq3>
meas_num = 0
pulse_num = -2
wl_ax = []
p_ax = []
peak_loc = []
peak_loc_wl = []
for i, d in enumerate(data):
    wl_ax.append(d["wavelengths"][pulse_num, meas_num, :])
    p_ax.append(d["powers"][pulse_num, meas_num, :])
    peak_loc.append(p_ax[i].argmax())
    peak_loc_wl.append(wl_ax[i][peak_loc[i]])
#|%%--%%| <fPNSjlkTq3|oBwVEKKK40>
# make fit of peak_loc_wl
pump_sep = np.array([40, 35, 30, 25])
y = peak_loc_wl
x = pump_sep
fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
fig, ax = plt.subplots()
ax.plot(x, y, "o", x, fit_fn(x), "--k")
ax.set_xlabel("Pump separation (nm)")
ax.set_ylabel("Peak location (nm)")
ax.set_title("Peak location vs pump separation")
