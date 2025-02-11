# %%
import pickle
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
from funcs.sweep_tisa_to_find_opt import sweep_tisa_multiple_pump_seps

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1618
pump2_start = 1604
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.2)
ando_pow_start = 0
# pump1_laser = PhotoneticsLaser(pump1_start, GPIB_address=10, power=6)
pump1_laser = AndoLaser(pump1_start, GPIB_address=24, power=8)
pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
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

verdi.power = 8
verdi.shutter = 0
time.sleep(0.5)
# verdi.shutter = 1

# %%
# L band only params
pump1_start = 1609
pump2_start = 1604
pump1_end = 1615
pump2_end = pump2_start
pump1_wls = np.arange(pump1_start, pump1_end + 1, 1)
pump2_wls = np.ones_like(pump1_wls) * pump2_start
# manual_pump_wl_list = [(1591, 1589)]
# p_wl_mean = np.mean([pump1_start, pump2_start])
pump_modes = ["21", "02"]
data_folders = [
    f"../data/4MSI/tisa_sweep_to_find_opt/pump_modes={pump_modes[0]}_equal_input_pump_p",
    f"../data/4MSI/tisa_sweep_to_find_opt/pump_modes={pump_modes[1]}_equal_input_pump_p",
]
data_folders = [
    f"../data/4MSI/tisa_sweep_to_find_opt/pump_modes={pump_modes[0]}_skewed_input_pump_p",
    f"../data/4MSI/tisa_sweep_to_find_opt/pump_modes={pump_modes[1]}_skewed_input_pump_p",
]
# data_folders = [data_folders[1]]
pump_stepsize = 1
total_pump_move_nm = pump1_end - pump1_start
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1
pump_wl_list = [
    (pump1_wl, pump2_wl) for pump1_wl, pump2_wl in zip(pump1_wls, pump2_wls)
]
# pump_wl_list = [
#     (pump1_start + pump_stepsize * i, pump2_start - pump_stepsize * i)
#     for i in range(num_pump_steps)
# ]

# pump_wl_list = sorted(pump_wl_list + manual_pump_wl_list, key=lambda x: x[0])
min_wavelength = min(pump_wl_list, key=lambda x: x[0])[0]
max_wavelength = max(pump_wl_list, key=lambda x: x[0])[0]
total_pump_move_nm = max_wavelength - min_wavelength
num_pump_steps = len(pump_wl_list)
start_wls = [934, 949]
pump_powers = np.array([[8, 8] for _ in range(num_pump_steps)])
pump_powers[3:, 1] = -6
# %%
# If 0, then only the tisa point from linear fit will be measured
for i in range(len(data_folders)):
    idler_side = "red"
    osa_params = {
        "res": 0.1,
        "sens": "SMID",
    }
    tisa_stepsize = 0.13
    num_tisa_steps = 30
    sampling_ratio = 0
    start_wl = start_wls[i]
    verdi_power = 9
    verdi.power = verdi_power
    num_sweep_reps = 1
    duty_cycles = [0.1]
    params = {
        "pump_powers": pump_powers,
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
    data_folder = data_folders[i]
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    importlib.reload(sys.modules["funcs.sweep_tisa_to_find_opt"])
    from funcs.sweep_tisa_to_find_opt import sweep_tisa_multiple_pump_seps

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
        initialize_tisa_properly=True,
        ipg_edfa=None,
        error_tolerance=0.15,
    )
# %%
pump1_wavelengths = np.arange(1609, 1617, 1)
pump2_wavelengths = np.ones_like(pump1_wavelengths) * 1604

spec_tmp = []
for p1wl, p2wl in zip(pump1_wavelengths, pump2_wavelengths):
    spectrum = np.vstack((np.linspace(0, 1), np.linspace(0, 1)))
    spec_tmp.append(spectrum)
specs = np.array(spec_tmp)
# %%  Save pump spectra
pump1_wavelengths = np.arange(1609, 1617, 1)
pump2_wavelengths = np.ones_like(pump1_wavelengths) * 1604
pump_powers = ((8, 8), (8, -6))
spectra = {"dc": 0.2}
osa.stop_sweep()
osa.sweeptype = "SGL"
for p1, p2 in pump_powers:
    spec_tmp = []
    pump1_laser.power = p1
    pump2_laser.power = p2
    for p1wl, p2wl in zip(pump1_wavelengths, pump2_wavelengths):
        pump1_laser.wavelength = p1wl
        pump2_laser.wavelength = p2wl
        time.sleep(0.2)
        osa.span = (p2wl - 2.5, p1wl + 2.5)
        osa.sweep()
        spectrum = np.vstack((osa.wavelengths, osa.powers))
        spec_tmp.append(spectrum)
    spectra[(p1, p2)] = spec_tmp

save_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\4MSI\pump_specs"
with open(os.path.join(save_dir, "pump_specs_edfa_wl_limit.pkl"), "wb") as f:
    pickle.dump(spectra, f)
# %% Simple save pump specs
save_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\4MSI\pump_specs"
osa.update_spectrum()
spectrum = np.vstack((osa.wavelengths, osa.powers))
with open(
    os.path.join(
        save_dir, "CW_throughAOM_high_P_EDFA_after_tap_1604_1618_KK_at_1618.pkl"
    ),
    "wb",
) as f:
    pickle.dump(spectrum, f)
# %%
