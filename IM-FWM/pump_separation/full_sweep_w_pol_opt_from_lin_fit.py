# %%
import numpy as np
from typing import List
import os
import time
import matplotlib.pyplot as plt
import sys
import pickle


sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser, TiSapphire
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from amonics_edfa import EDFA
from pol_cons import ThorlabsMPC320
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA
from misc import PM
from funcs.sweep_pumps_around_opt import (
    sweep_w_pol_opt_based_on_linear_fit,
    turn_off_all_lasers,
)

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
ando1_start = 1577
ando2_start = 1563
ipg_edfa = IPGEDFA(connection_mode="GPIB", GPIB_address=17)
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.1)
ando_pow_start = 0
ando1_laser = AndoLaser(ando1_start, GPIB_address=24, power=8)
ando2_laser = AndoLaser(ando2_start, GPIB_address=23, power=8)
time.sleep(0.1)
ando1_laser.laser_on()
ando2_laser.laser_on()
tisa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    960,
    990,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="RPT",
)

verdi.power = 6
verdi.shutter = 0
time.sleep(0.5)
verdi.shutter = 1
ipg_edfa.status = 1
# %% Pump spectra
data_folder = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\C_plus_L_band\cleo"
osa_res = 0.2
ando1_start = 1577
ando2_start = 1563
ando1_end = 1607
ando2_end = 1533
pump_stepsize = 2.5
total_pump_move_nm = ando1_end - ando1_start
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1

pump_wl_list = [
    (ando1_start + pump_stepsize * i, ando2_start - pump_stepsize * i)
    for i in range(num_pump_steps)
]
osa.resolution = osa_res
osa.sweeptype = "SGL"
spec_dict = {}
for pump_wl_pair in pump_wl_list:
    ando1_laser.wavelength = pump_wl_pair[0]
    ando2_laser.wavelength = pump_wl_pair[1]
    osa.span = (pump_wl_pair[1] - 8, pump_wl_pair[0] + 8)
    osa.sweep()
    wavelengths = osa.wavelengths
    powers = osa.powers
    spectrum = np.vstack((wavelengths, powers))
    spec_dict[pump_wl_pair] = spectrum
with open(os.path.join(data_folder, "pump_spectra.pkl"), "wb") as f:
    pickle.dump(spec_dict, f)

# %% Sweep TISA measurements

phase_matching_approx_fit_loc = "./processing/fits/C_plus_L_band/linear_fit.txt"
phase_matching_approx_fit = np.poly1d(
    np.loadtxt(phase_matching_approx_fit_loc, delimiter=",")
)
#  Initial setup settings
ando1_start = 1577
ando2_start = 1563
ando1_end = 1607
ando2_end = 1533

# L band only params
ando1_start = 1592.5
ando2_start = 1587.5
ando1_end = 1610
ando2_end = 1570
manual_pump_wl_list = [(1591, 1589)]

pump_stepsize = 2.5
total_pump_move_nm = ando1_end - ando1_start
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1
pump_wl_list = [
    (ando1_start + pump_stepsize * i, ando2_start - pump_stepsize * i)
    for i in range(num_pump_steps)
]
pump_wl_list = sorted(pump_wl_list + manual_pump_wl_list, key=lambda x: x[0])

min_wavelength = min(pump_wl_list, key=lambda x: x[0])[0]
max_wavelength = max(pump_wl_list, key=lambda x: x[0])[0]
total_pump_move_nm = max_wavelength - min_wavelength
num_pump_steps = len(pump_wl_list)

pump1_wl_stepsize = 0.02
pump1_wl_delta = 0.2
num_pump1_steps = int(pump1_wl_delta / pump1_wl_stepsize) * 2 + 1
pump2_wl_stepsize = 0.03
pump2_wl_delta = 0.0
num_pump2_steps = int(pump2_wl_delta / pump2_wl_stepsize) * 2 + 1
auto_pol_opt = True
manual_pol_opt = False

# If 0, then only the tisa point from linear fit will be measured
idler_side = "red"
duty_cycles = [0.1, 0.2, 0.5, 1]
osa_params = {
    "res": 0.05,
    "sens": "SMID",
}
sampling_ratio = 0
verdi_power = 6
verdi.power = verdi_power
num_sweep_reps = 5
pol_opt_dc = 0.1
params = {
    "ando1_start": ando1_start,
    "ando2_start": ando2_start,
    "pump_stepsize": pump_stepsize,
    "total_pump_move_nm": total_pump_move_nm,
    "num_pump_steps": num_pump_steps,
    "idler_side": idler_side,
    "duty_cycles": duty_cycles,
    "osa_params": osa_params,
    "verdi_power": verdi_power,
    "num_sweep_reps": num_sweep_reps,
    "pol_opt_dc": pol_opt_dc,
    "sampling_ratio": sampling_ratio,
    "pump1_wl_stepsize": pump1_wl_stepsize,
    "pump1_wl_delta": pump1_wl_delta,
    "pump2_wl_stepsize": pump2_wl_stepsize,
    "pump2_wl_delta": pump2_wl_delta,
    "phase_match_fit": phase_matching_approx_fit_loc,
    "pulse_freq": pulse_freq,
}
data_folder = (
    "./data/C_plus_L_band/cleo/pol_opt_often/l_band_only/sweep_long_p_around_opt"
)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


# %%  sweep pump wavelength finely around tisa opt from linear fit
sweep_w_pol_opt_based_on_linear_fit(
    pump_wl_list,
    params,
    osa_params,
    num_pump_steps,
    ando1_laser,
    ando2_laser,
    tisa,
    osa,
    pico,
    pol_con1,
    pol_con2,
    arduino,
    data_folder,
    pol_opt_method="auto",
    ipg_edfa=ipg_edfa,
)

turn_off_all_lasers([ando1_laser, ando2_laser], verdi, [ipg_edfa])
# %%
data_loc = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\C_plus_L_band\cleo\pol_opt_auto\pump_wl=(1589.5, 1550.5)_cband_pump_sweep_around_opt.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)
plt.plot(data["spectra"][0.1][0, 0, 2, 1, :])
