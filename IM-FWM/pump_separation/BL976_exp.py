import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
import sys

sys.path.append(
    r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\InstrumentControl"
)
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from OSA_control import OSA
from laser_control import laser
from amonicsEDFA import EDFA
from picoscope2000 import PicoScope2000a

plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
GPIB_val = 0
# |%%--%%| <TXhj36piLN|lvTPpyhVck>
ando1_start = 1570
ando2_start = 1563
duty_cycle = [0.1, 0.5, 1]
pulse_freq = 1 * 10**5
# edfa1 = EDFA("COM7")
# edfa2 = EDFA("COM5")
pico = PicoScope2000a()
pico.awg.set_square_wave(pulse_freq, duty_cycle[-1])
ando_pow_start = 0
ando1 = laser("ando", ando1_start, power=ando_pow_start, GPIB_num=GPIB_val)
ando2 = laser("ando2", ando2_start, power=ando_pow_start, GPIB_num=GPIB_val)
time.sleep(0.1)
ando1.laserON()
ando2.laserON()
osa = OSA(
    ando2_start - 2,
    ando1_start + 2,
    resolution=0.1,
    GPIB_num=[GPIB_val, 19],
    sweeptype="SGL",
)
# |%%--%%| <lvTPpyhVck|pI3LO235ZB>
data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/BL976_signal"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
step_size = 0.05
osa.set_res(0.1)
total_nm_window = 5
start_ando1 = ando1.get_wavelength()
start_ando2 = ando2.get_wavelength()
osa.get_spectrum()
num_points_per_spectrum = len(osa.wavelengths)
num_points = int(total_nm_window / step_size)
data = {"Pump1:": np.array([start_ando1 + i * step_size for i in range(num_points)]), "Pump2": start_ando2, "Wavelengths": osa.wavelengths, "Powers": np.zeros((num_points, num_points_per_spectrum))}
start_time = time.time()
for i in range(num_points):
    ando1.set_wavelength(start_ando1 + i * step_size)
    time.sleep(1)
    osa.sweep()
    power = osa.powers
    data["Powers"][i, :] = power
    print("Progress: {:.2f}%".format((i + 1) / num_points * 100))
    print("Time spent: {:.2f} s".format(time.time() - start_time))
with open(data_folder + "/long_pwl_sweep_no_polopt.pkl", "wb") as f:
    pickle.dump(data, f)
#|%%--%%| <pI3LO235ZB|Hu5BRqtjRU>
# measure same spectrum multiple times
data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/BL976_signal/pol_check_sig_opt_for_max_power"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
num_spectra = 10
osa.set_res(0.1)
osa.sweep()
osa.get_spectrum()
num_points_per_spectrum = len(osa.wavelengths)
pump1_wl = ando1.get_wavelength()
pump2_wl = ando2.get_wavelength()
data = {"Pump1:": pump1_wl, "Pump2": pump2_wl, "Wavelengths": osa.wavelengths, "Powers": np.zeros((num_spectra, num_points_per_spectrum))}
pump_sep = pump2_wl - pump1_wl
start_time = time.time()
for i in range(num_spectra):
    osa.sweep()
    power = osa.powers
    data["Powers"][i, :] = power
    print("Progress: {:.2f}%".format((i + 1) / num_spectra * 100))
    print("Time spent: {:.2f} s".format(time.time() - start_time))
with open(data_folder + f"/pol_min_pump_sep{pump_sep:.2f}=.pkl", "wb") as f:
    pickle.dump(data, f)
#|%%--%%| <R9cECe62Bf|6MKFewec1z>
data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/BL976_signal"
pump1_wl = ando1.get_wavelength()
pump2_wl = ando2.get_wavelength()
data = {"Pump1:": pump1_wl, "Pump2": pump2_wl, "Wavelengths": osa.wavelengths, "Powers": osa.powers}
with open(data_folder + f"/0nm_sweep_w_pol_change.pkl", "wb") as f:
    pickle.dump(data, f)
#|%%--%%| <Hu5BRqtjRU|R9cECe62Bf>
# measure pump wavelength multiple times
data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/BL976_signal"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
step_size = 0.05
osa.set_res(0.1)
total_nm_window = 25
start_ando1 = ando1.get_wavelength()
start_ando2 = ando2.get_wavelength()
osa.get_spectrum()
num_points_per_spectrum = len(osa.wavelengths)
num_points = int(total_nm_window / step_size)
data = {"Pump1:": start_ando1, "Pump2": np.array([start_ando2 + i * step_size for i in range(num_points)]), "Wavelengths": osa.wavelengths, "Powers": np.zeros((num_points, num_points_per_spectrum))}
start_time = time.time()
for i in range(num_points):
    new_wl = start_ando2 + i * step_size
    ando2.set_wavelength(new_wl)
    osa.set_span(new_wl - 1, new_wl + 1)
    time.sleep(1)
    osa.sweep()
    power = osa.powers
    data["Powers"][i, :] = power
    print("Progress: {:.2f}%".format((i + 1) / num_points * 100))
    print("Time spent: {:.2f} s".format(time.time() - start_time))
with open(data_folder + f"/pump_specs.pkl", "wb") as f:
    pickle.dump(data, f)
