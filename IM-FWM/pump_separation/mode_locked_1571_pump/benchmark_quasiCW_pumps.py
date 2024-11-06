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
pump1_start = 1608
pump2_start = 1604
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.2)
ando_pow_start = 0
# pump1_laser = PhotoneticsLaser(pump1_start, GPIB_address=10, power=6)
pump1_laser = AndoLaser(pump1_start, GPIB_address=23, power=8)
pump2_laser = AndoLaser(pump2_start, GPIB_address=24, power=8)
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
    trace="a",
)

# verdi.power = 6
# verdi.shutter = 0
time.sleep(0.5)
# verdi.shutter = 1

# %%
# Sweep to find optimal phase matching, from quasi CW pumps
pump1_start = 1571
pump2_start = 1563
pump1_end = 1571
pump2_end = 1533
pump_stepsize = 2.5
total_pump_move_nm = pump2_end - pump2_start
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1
pump2_wl_list = np.arange(pump2_start, pump2_end - 0.1, -pump_stepsize)
pump_wl_list = [(pump1_start, p2wl) for p2wl in pump2_wl_list]

min_wavelength = pump2_end
max_wavelength = pump1_start
total_pump_move_nm = max_wavelength - min_wavelength
num_pump_steps = len(pump2_wl_list)

# If 0, then only the tisa point from linear fit will be measured
idler_side = "red"
osa_params = {
    "res": 0.05,
    "sens": "SMID",
}
tisa_stepsize = 0.1
num_tisa_steps = 30
sampling_ratio = 0
start_wl = 982
verdi_power = 8
verdi.power = verdi_power
num_sweep_reps = 1
duty_cycles = [1]
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
data_folder = f"../data/modelocked_1571_pump"
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
# %%  Save pump spectra
pump1_wavelengths = np.arange(1577, 1608, 2.5)
pump2_wavelengths = np.arange(1563, 1532, -2.5)
spectra = {"dc": 0.2}
osa.stop_sweep()
osa.sweeptype = "SGL"
for p1wl, p2wl in zip(pump1_wavelengths, pump2_wavelengths):
    pump1_laser.wavelength = p1wl
    pump2_laser.wavelength = p2wl
    time.sleep(0.2)
    osa.span = (p2wl - 2.5, p1wl + 2.5)
    osa.sweep()
    spectrum = np.vstack((osa.wavelengths, osa.powers))
    spectra[(np.round(p1wl, 1), np.round(p2wl, 1))] = spectrum
spectra["pump_sep"] = np.abs(pump1_wavelengths - pump2_wavelengths)
save_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\C_plus_L_band\pump_spectra"
with open(os.path.join(save_dir, "spectra.npy"), "wb") as f:
    np.save(f, spectra)
# %%
# %%
# data_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\"
osa.stop_sweep()
osa.sweeptype = "SGL"
p1_wl = pump1_laser.wavelength
p2_wl = pump1_laser.wavelength

for i in range(60):
    tisa.delta_wl_nm(0.2)
    osa.sweep()
    osa.update_spectrum()

    # spectrum = np.vstack((osa.wavelengths, osa.powers))
    # np.savetxt(
    #     os.path.join(
    #         data_dir,
    #         f"spec_p2wl={pump2_laser.wavelength:.1f}_p1pm_wl={p1_wl:.2f}_iter={i}.csv",
    #     ),
    #     spectrum,
    #     fmt="%f",
    #     delimiter=",",
    # )
