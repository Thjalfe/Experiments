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
from laser_control import AndoLaser, TiSapphire, PhotoneticsLaser
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons
from arduino_pm import ArduinoADC
from funcs.sweep_tisa_to_find_opt import sweep_tisa_multiple_pump_seps

importlib.reload(sys.modules["funcs.sweep_tisa_to_find_opt"])

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1580
pump2_start = 1570
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=60)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=60)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.2)
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
    sweeptype="RPT",
)

verdi.power = 6
verdi.shutter = 0
time.sleep(0.5)
verdi.shutter = 1


# %%
def pump_freq_diff_init(pump_wl: float, wl_dist: float, c=299792458):
    pump_wl = pump_wl * 10**-9
    wl_dist = wl_dist * 10**-9
    pump_freq_1 = c / pump_wl
    pump_freq_2 = c / (pump_wl + wl_dist)
    return pump_freq_1 - pump_freq_2


def new_pump_wl_from_freq_diff(
    pump_wl_start: float,
    freq_diff: float,
    c=299792458,
    longer_or_shorter="longer",
    rounding=2,
):
    pump_wl_start = pump_wl_start * 10**-9
    pump_freq_start = c / pump_wl_start
    if longer_or_shorter == "longer":
        pump_freq_end = pump_freq_start - freq_diff
    elif longer_or_shorter == "shorter":
        pump_freq_end = pump_freq_start + freq_diff
    pump_wl_end = c / pump_freq_end
    return np.round(pump_wl_end * 10**9, rounding)


pump_wl_short_start = 1570
pump_wl_short_stepsize = 3
pump_wl_short_end = 1610
setup_wl_limits = (1533, 1610)
init_dist_between_pumps = 10
freq_diff = pump_freq_diff_init(pump_wl_short_start, init_dist_between_pumps)
pump2_wl_array = np.arange(
    pump_wl_short_start, pump_wl_short_end, pump_wl_short_stepsize
)
pump1_wl_array = np.array(
    [
        new_pump_wl_from_freq_diff(wl, freq_diff, longer_or_shorter="longer")
        for wl in pump2_wl_array
    ]
)
mask1 = (pump1_wl_array >= setup_wl_limits[0]) & (pump1_wl_array <= setup_wl_limits[1])
mask2 = (pump2_wl_array >= setup_wl_limits[0]) & (pump2_wl_array <= setup_wl_limits[1])
combined_mask = mask1 & mask2
pump1_wl_array = pump1_wl_array[combined_mask]
pump2_wl_array = pump2_wl_array[combined_mask]
pump1_wl_array = pump1_wl_array[::-1]
pump2_wl_array = pump2_wl_array[::-1]

pump_wl_list = np.array([pump1_wl_array, pump2_wl_array]).T.tolist()
idler_side = "red"
osa_params = {
    "res": 0.05,
    "sens": "SMID",
}
tisa_stepsize = 0.1
num_tisa_steps = 40
sampling_ratio = 5
start_wl = 963.5
verdi_power = 6
verdi.power = verdi_power
num_sweep_reps = 1
duty_cycles = [0.2]
params = {
    "pump_wl_list": pump_wl_list,
    "pump1_start": pump1_start,
    "pump2_start": pump2_start,
    "pump_stepsize": pump_wl_short_stepsize,
    "total_pump_move_nm": pump1_wl_array[-1] - pump1_wl_array[0],
    "num_pump_steps": len(pump1_wl_array),
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

data_folder = f"../data/tisa_sweep_to_find_opt/moving_pump_mean/pump_wl_dist={init_dist_between_pumps}nm"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
pump_idx = -1
pump1_wl = pump1_wl_array[pump_idx]
pump2_wl = pump2_wl_array[pump_idx]
pump1_laser.wavelength = pump1_wl
pump2_laser.wavelength = pump2_wl
# %%
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
