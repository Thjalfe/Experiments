# %%
import numpy as np
from typing import List
import os
import time
import matplotlib.pyplot as plt
import sys
import pickle

os.chdir(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\distant_bs")
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
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA
from misc import PM


plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1582.5
seed_start = 1597.5
# pump2_start = 1589
# ipg_edfa = IPGEDFA(connection_mode="GPIB", GPIB_address=17)
ipg_edfa = None
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 1)
ando_pow_start = 0
pump1_laser = AndoLaser(pump1_start, GPIB_address=24, power=8)
pump2_laser = PhotoneticsLaser(seed_start, GPIB_address=10, power=6)
# pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
# pump2_laser = PhotoneticsLaser(pump2_start, GPIB_address=10, power=8)
time.sleep(0.1)
pump1_laser.enable()
pump2_laser.enable()
# pump2_laser.enable()
tisa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    960,
    990,
    resolution=2,
    GPIB_address=19,
    sweeptype="SGL",
)
# pm = PM()

# verdi.power = 6
# verdi.shutter = 0
time.sleep(0.5)


# verdi.shutter = 1
# ipg_edfa.status = 1
# %%
def save_spectrum(
    osa: OSA,
    data_path: str,
    file_name: str,
    signal_power: float | str,
    resolution: float = 0.1,
    sensitivity: str = "SMID",
):
    osa.stop_sweep()
    osa.sweeptype = "SGL"
    osa.resolution = resolution
    osa.sensitivity = sensitivity
    osa.sweep()
    spectrum = np.vstack((osa.wavelengths, osa.powers))
    data = {"spectrum": spectrum, "signal_power": signal_power}
    with open(os.path.join(data_path, file_name), "wb") as f:
        pickle.dump(data, f)
    return spectrum


def save_timetrace(
    osa: OSA,
    data_path: str,
    file_name: str,
):
    osa.stop_sweep()
    osa.update_spectrum()
    timetrace = np.vstack((np.linspace(0, 50, len(osa.powers)), osa.powers))
    np.savetxt(
        os.path.join(data_path, file_name),
        timetrace,
        delimiter=",",
    )
    return timetrace


def set_sig_idler_markers(osa: OSA, idx_dist_from_pump_min=20):
    osa.update_spectrum()
    pump_power = np.max(osa.powers)
    pump_loc = np.argmax(osa.powers)
    pump_wl = osa.wavelengths[pump_loc]
    idler_loc = (
        np.argmax(osa.powers[pump_loc + idx_dist_from_pump_min :])
        + pump_loc
        + idx_dist_from_pump_min
    )
    idler_power = osa.powers[idler_loc]
    idler_wl = osa.wavelengths[idler_loc]
    osa.set_wavelength_marker(1, pump_wl)
    osa.set_wavelength_marker(2, idler_wl)
    osa.set_power_marker(3, pump_power)
    osa.set_power_marker(4, idler_power)


pump1_start = 1582.5
pump2_start = 1597.5
pump1_laser.wavelength = pump1_start
pump2_laser.wavelength = pump2_start
base_path = os.getcwd()
data_path = os.path.join(
    base_path,
    rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\distant_bs\data\tisa_pump\{pump1_start:.1f}nm_{pump2_start:.1f}nm_pumps",
)
if not os.path.exists(data_path):
    os.makedirs(data_path)
# %%
save_spectrum(
    osa,
    data_path,
    f"pump1wl={pump1_start:.1f}_pump2wl={pump2_start:.1f}_idler_spec_filtered.csv",
    4.76,
)
# %%
save_spectrum(
    osa,
    data_path,
    f"distant_bs_for_comp_pump1wl={pump1_start:.1f}_pump2wl={pump2_start:.1f}.csv",
    "Read on spectrum",
)
# %% old fit for mean pump wl 1590 nm
data_loc = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\processing\fits\mean_pumpwl_1590.0nm.txt"
data = np.loadtxt(data_loc, delimiter=",")
# it is a numpy polynomial fit
data_fit = np.poly1d(data)
# %% optimize pol cons
optimize_multiple_pol_cons(arduino, pol_con1, pol_con2, tolerance=0.5)
# %% load some data
data_path = (
    r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\distant_bs\data\tisa_pump"
)
filename = rf"{pump1_start:.1f}nm_{pump2_start:.1f}nm_pumps\distant_bs_for_comp_pump1wl={pump1_start:.1f}_pump2wl={pump2_start:.1f}.csv"
data = np.loadtxt(os.path.join(data_path, filename), delimiter=",")
