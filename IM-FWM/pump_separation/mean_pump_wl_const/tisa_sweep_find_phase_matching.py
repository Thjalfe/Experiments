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
from laser_control import AndoLaser, TiSapphire
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from pol_cons import ThorlabsMPC320
from arduino_pm import ArduinoADC
from funcs.sweep_tisa_to_find_opt import sweep_tisa_multiple_pump_seps

importlib.reload(sys.modules["funcs.sweep_tisa_to_find_opt"])

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1591
pump2_start = 1589
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.2)
ando_pow_start = 0
pump1_laser = AndoLaser(pump1_start, GPIB_address=24, power=8)
pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
time.sleep(0.1)
pump1_laser.laser_on()
pump2_laser.laser_on()
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

# %%
# L band only params
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
idler_side = "red"
osa_params = {
    "res": 0.05,
    "sens": "SMID",
}
tisa_stepsize = 0.1
num_tisa_steps = 30
sampling_ratio = 0
start_wl = 972
verdi_power = 6
verdi.power = verdi_power
num_sweep_reps = 1
duty_cycles = [0.2]
params = {
    "pump_wl_list": pump_wl_list,
    "pump1_start": pump1_start,
    "pump2_start": pump2_start,
    "pump_stepsize": pump_stepsize,
    "total_pump_move_nm": total_pump_move_nm,
    "num_pump_steps": num_pump_steps,
    "idler_side": idler_side,
    "osa_params": osa_params,
    "verdi_power": verdi_power,
    "num_sweep_reps": num_sweep_reps,
    "sampling_ratio": sampling_ratio,
    "pulse_freq": pulse_freq,
    "mean_p_wl": np.mean([pump1_start, pump2_start]),
    "start_wl": start_wl,
    "tisa_stepsize": tisa_stepsize,
    "num_tisa_steps": num_tisa_steps,
    "duty_cycles": duty_cycles,
}
data_folder = f"../data/tisa_sweep_to_find_opt/pump_wl_mean_{p_wl_mean}"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
sweep_tisa_multiple_pump_seps(
    params,
    osa_params,
    pump1_laser,
    pump2_laser,
    verdi,
    tisa,
    osa,
    pico,
    data_folder,
    ipg_edfa=None,
)
