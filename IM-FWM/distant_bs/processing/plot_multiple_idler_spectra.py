# %%
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle

data_loc = "../data/wide_tisa_sweeps/toptica/tisa_sweep_pump1wl=1605_pump2wl=1029.4_50%toptica.pkl"
import pandas as pd

data = pd.read_pickle(data_loc)
data_keys = [key for key in data.keys() if isinstance(key, (int, float))]
data_keys.sort()
# %%
fig, ax = plt.subplots()
# Plotting each spectrum
data_change = copy.deepcopy(data)
for key in data_keys:
    spectrum = data_change[key]
    spectrum[1][spectrum[1] < -77] = np.nan
    ax.plot(spectrum[0], spectrum[1], label=f"Wavelength: {key} nm")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity")
# ax.set_title("Spectra for Various Laser Wavelengths")
# ax.legend()
# %%
from scipy.signal import find_peaks

# Dictionary to store the highest prominence for each spectrum
peak_prominences = {}

# Analyze each spectrum
for key in data_keys:
    spectrum = data_change[key]
    spectrum[1][spectrum[1] < -77] = np.nan
    wavelengths = spectrum[0]
    intensities = spectrum[1]

    # Finding peaks
    peaks, properties = find_peaks(
        intensities, prominence=1
    )  # minimum prominence set to 1 for filtering minor peaks
    if peaks.size > 0:
        # Find the peak with the highest prominence
        max_prominence = max(properties["prominences"])
        peak_prominences[key] = max_prominence

# Sort the spectra based on the highest prominence
sorted_keys = sorted(peak_prominences, key=peak_prominences.get, reverse=True)[
    :10
]  # top 10 keys

# Plot the top 10 spectra
plt.figure(figsize=(15, 10))
for key in sorted_keys:
    spectrum = data_change[key]
    spectrum[1][spectrum[1] < -77] = np.nan
    plt.plot(spectrum[0], spectrum[1], label=f"Wavelength: {key} nm")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Top 10 Spectra with Most Prominent Peaks")
plt.legend()

# Show the plot
plt.show()
