# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob

data_dir = r"../data/sweep_without_seed"
files = glob.glob(os.path.join(data_dir, "*.pkl"))

data = []
for file in files:
    data.append(pickle.load(open(file, "rb")))
# np.nan every value in :, 1, : where the power is less than -85
for d in data:
    d["spectra"][d["spectra"] < -85] = np.nan
# %%


def plot_spectra_naive_unseeded_sweep(data_dict, plot_every_nth_pump_wl=1):
    pump_wls = data_dict["pump_wls"][::plot_every_nth_pump_wl]
    spectra = data_dict["spectra"][::plot_every_nth_pump_wl]
    for idx_p, pump_wl in enumerate(pump_wls):
        plt.plot(spectra[idx_p][0], spectra[idx_p][1], label=f"{pump_wl:.1f} nm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power (dBm)")


plot_spectra_naive_unseeded_sweep(data[1], plot_every_nth_pump_wl=1)
