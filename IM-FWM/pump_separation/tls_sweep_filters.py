# %%
import pickle
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys

sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
laser_wavelength = 1610
# pump2_start = 1604
ando_pow_start = 0
# pump1_laser = PhotoneticsLaser(pump1_start, GPIB_address=10, power=6)
# laser = AndoLaser(laser_wavelength, GPIB_address=24, power=8)
# pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
time.sleep(0.1)
# laser.enable()
# pump2_laser.enable()
osa = OSA(
    1570,
    1610,
    resolution=0.1,
    GPIB_address=19,
    sweeptype="RPT",
)
# %%
wl_start = 1600
wl_stop = 1610
dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\characterization\filters"
filename = f"edmund1200SPF_{int(wl_start)}_{int(wl_stop)}.json"
stepsize = 0.5
l_band_wavelengths = np.arange(wl_start, wl_stop + stepsize, stepsize)
spectrum_dict = {wl: [] for wl in l_band_wavelengths}
osa.sweeptype = "SGL"
osa.sensitivity = "SHI1"
osa.stop_sweep()
for wl in l_band_wavelengths:
    laser.wavelength = wl
    osa.span = (wl - 2.5, wl + 2.5)
    osa.sweep()
    wavelengths = osa.wavelengths
    powers = osa.powers
    spectrum_dict[wl] = np.array(list(zip(wavelengths, powers)))
with open(os.path.join(dir, filename), "wb") as f:
    pickle.dump(spectrum_dict, f)

# %%

wl_start = 1520
wl_stop = 1600
dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\characterization\filters"
filename = f"edmund1300SPF2_{int(wl_start)}_{int(wl_stop)}.json"
# filename = f"reference_{int(wl_start)}_{int(wl_stop)}.json"
osa.stop_sweep()
osa.TLS = 0
osa.sweeptype = "SGL"
osa.resolution = 1
osa.span = (wl_start, wl_stop)
osa.sensitivity = "SHI1"
osa.TLS = 1
# %%
filename = f"edmund1200SPF_20degtilt_{int(wl_start)}_{int(wl_stop)}.json"
# filename = f"reference_{int(wl_start)}_{int(wl_stop)}.json"
# filename = f"edmund1300SPF1_{int(wl_start)}_{int(wl_stop)}.json"
osa.sweep()
# %%
osa.update_spectrum()
wavelengths = osa.wavelengths
powers = osa.powers
spectrum = np.array(list(zip(wavelengths, powers)))
with open(os.path.join(dir, filename), "wb") as f:
    pickle.dump(spectrum, f)
