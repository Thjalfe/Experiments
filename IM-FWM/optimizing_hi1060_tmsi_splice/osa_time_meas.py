# %%
import numpy as np
from typing import List
import os
import time
import matplotlib.pyplot as plt
import sys
import importlib

sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import TiSapphire

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
tisa = TiSapphire(5)
osa = OSA(
    960,
    990,
    resolution=0.5,
    GPIB_address=19,
    sweeptype="RPT",
    trace="a",
)

# %%

save_data_dir = "./osa_time_meas/tisa_wl=950nm_full_setup"
# save_data_dir = "./osa_time_meas/tisa_wl=950nm_2msi-len=100m"
num_meas = 11
meas_time = 50
total_time = num_meas * meas_time
meas_name = f"meas3_2refuse_{total_time:.0f}sec"
meas_name = f"meas3_{total_time:.0f}sec"
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
osa.span = 950
osa.sweeptype = "SGL"
pow_vs_time = []
time_array = []
for i in range(num_meas):
    osa.sweep()
    powers = osa.powers
    time_ax = np.linspace(i * meas_time, (i + 1) * meas_time, len(powers))
    pow_vs_time.append(powers)
    time_array.append(time_ax)
    print(f"Measurement {i + 1}/{num_meas} done.")
pow_time_reshaped = np.array(pow_vs_time).reshape(-1)
time_reshaped = np.array(time_array).reshape(-1)
time_and_power = np.vstack((time_reshaped, pow_time_reshaped))
np.savetxt(
    os.path.join(save_data_dir, meas_name + ".csv"), time_and_power, delimiter=","
)
osa.sweeptype = "RPT"
osa.sweep()
