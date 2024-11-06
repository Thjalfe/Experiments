import glob
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

data_dir = "../data/LPG_T_3_1600/power_char"
data_files = glob.glob(data_dir + "/*.pkl")
output_file = [f for f in data_files if "output" in f][0]
input_file = [f for f in data_files if "input" in f][0]
with open(output_file, "rb") as f:
    output_data = np.load(f, allow_pickle=True)
with open(input_file, "rb") as f:
    input_data = np.load(f, allow_pickle=True).item()
tisa_wl = input_data["tisa"][0][1:]
tisa_in = input_data["tisa"][1][1:]
tisa_out = output_data["tisa"][1][1:]
pump_wl = input_data["pumps"][0]
pump_in = input_data["pumps"][1]
pump_out = output_data["pumps"][1]

fig, ax = plt.subplots(2, 1)
ax[0].plot(tisa_wl, tisa_in - tisa_out)
ax[1].plot(pump_wl, pump_in - pump_out)
ax[0].set_title("TISA")
ax[1].set_title("Pump")
ax[0].set_xlabel("Wavelength (nm)")
ax[0].set_ylabel("Loss through fiber components")
ax[1].set_ylabel("LLoss through fiber component (dB)")

plt.show()
