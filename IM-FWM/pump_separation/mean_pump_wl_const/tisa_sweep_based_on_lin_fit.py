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
from laser_control import AndoLaser, TiSapphire, PhotoneticsLaser
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from amonics_edfa import EDFA
from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons, Agilent11896A
import importlib
from pump_separation.funcs.sweep_tisa_around_opt import (
    sweep_w_pol_opt_based_on_linear_fit,
)
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA
from misc import PM

importlib.reload(sys.modules["pump_separation.funcs.sweep_tisa_around_opt"])

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1591
pump2_start = 1589
# ipg_edfa = IPGEDFA(connection_mode="GPIB", GPIB_address=17)
ipg_edfa = None
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.5)
ando_pow_start = 0
pump1_laser = AndoLaser(pump1_start, GPIB_address=24, power=8)
pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
# pump2_laser = PhotoneticsLaser(pump2_start, GPIB_address=10, power=8)
time.sleep(0.1)
pump1_laser.enable()
pump2_laser.enable()
tisa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    960,
    990,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="SGL",
)
# pm = PM()

# verdi.power = 6
# verdi.shutter = 0
time.sleep(0.5)
# verdi.shutter = 1
# ipg_edfa.status = 1


# %% Sweep TISA measurements

phase_matching_approx_fit_loc = "../processing/fits/mean_pumpwl_1570.0nm.txt"
phase_matching_approx_fit = np.poly1d(
    np.loadtxt(phase_matching_approx_fit_loc, delimiter=",")
)
#  Initial setup settings
pump1_start = 1592.5
pump2_start = 1587.5
pump1_end = 1610
pump2_end = 1570
manual_pump_wl_list = [(1591, 1589)]
p_wl_mean = np.mean([pump1_start, pump2_start])

pump_stepsize = 2.5
total_pump_move_nm = pump1_end - pump1_start
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1
pump_wl_list = [
    (pump1_start + pump_stepsize * i, pump2_start - pump_stepsize * i)
    for i in range(num_pump_steps)
]
pump_wl_list = sorted(pump_wl_list + manual_pump_wl_list, key=lambda x: x[0])

min_wavelength = min(pump_wl_list, key=lambda x: x[0])[0]
max_wavelength = max(pump_wl_list, key=lambda x: x[0])[0]
total_pump_move_nm = max_wavelength - min_wavelength
num_pump_steps = len(pump_wl_list)

# If 0, then only the tisa point from linear fit will be measured
tisa_dist_either_side_of_peak = 0.5
tisa_stepsize = 0.1
num_tisa_steps = int(tisa_dist_either_side_of_peak / tisa_stepsize) * 2
idler_side = "red"
duty_cycles = [0.1, 0.2, 0.5, 1]
osa_params = {
    "res": 0.05,
    "sens": "SMID",
}
verdi_power = 6
verdi.power = verdi_power
num_sweep_reps = 3
pol_opt_dc = 0.1
sampling_ratio = 0
params = {
    "pump1_start": pump1_start,
    "pump2_start": pump2_start,
    "pump_stepsize": pump_stepsize,
    "total_pump_move_nm": total_pump_move_nm,
    "num_pump_steps": num_pump_steps,
    "pump_wl_list": pump_wl_list,
    "tisa_dist_either_side_of_peak": tisa_dist_either_side_of_peak,
    "tisa_stepsize": tisa_stepsize,
    "num_tisa_steps": num_tisa_steps,
    "idler_side": idler_side,
    "duty_cycles": duty_cycles,
    "osa_params": osa_params,
    "verdi_power": verdi_power,
    "num_sweep_reps": num_sweep_reps,
    "phase_match_fit": phase_matching_approx_fit,
    "pulse_freq": pulse_freq,
    "mean_p_wl": np.mean(pump_wl_list[0]),
    "pol_opt_dc": pol_opt_dc,
    "sampling_ratio": sampling_ratio,
}
data_folder = f"../data/sweep_multiple_separations_w_polopt/pol_opt_auto/tisa_sweep_around_opt/mean_p_wl={params['mean_p_wl']}"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
# %%
sweep_w_pol_opt_based_on_linear_fit(
    params,
    osa_params,
    pump1_laser,
    pump2_laser,
    verdi,
    tisa,
    osa,
    pico,
    pol_con1,
    pol_con2,
    arduino,
    data_folder,
    ipg_edfa,
)
# %% DISTANT BE SECTION
phase_matching_approx_fit_loc = "../processing/fits/mean_pumpwl_1590.0nm.txt"
phase_matching_approx_fit = np.poly1d(
    np.loadtxt(phase_matching_approx_fit_loc, delimiter=",")
)
# %%
p1_wl = 1589
s_wl = 1591
pump1_laser.wavelength = p1_wl
pump2_laser.wavelength = s_wl
osa.stop_sweep()
osa.sweeptype = "SGL"
osa.sweep()
osa.save(
    rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\distant_bs\data\1589_1591\p1={p1_wl}_s={s_wl}_short_wl_distant_bs"
)
