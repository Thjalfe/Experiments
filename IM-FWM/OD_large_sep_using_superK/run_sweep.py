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
from laser_control import AndoLaser, TiSapphire
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1610
# arduino = ArduinoADC("COM11")
# pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
# pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
verdi = VerdiLaser(com_port=12)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 1)
ando_pow_start = 0
# pump1_laser = PhotoneticsLaser(pump1_start, GPIB_address=10, power=6)
telecom_pump = AndoLaser(pump1_start, GPIB_address=24, power=8)
time.sleep(0.1)
telecom_pump.enable()
tisa = TiSapphire(5)
short_osa = OSA(
    960,
    990,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="RPT",
)
long_osa = OSA(
    1600,
    2000,
    resolution=0.05,
    GPIB_address=17,
    sweeptype="RPT",
)


# %%  SECTION FOR GETTING SPECTRA OF TISA PUMP AND SUPERK SPECTRA
def sweep_and_save_spectrum(
    osa: OSA, span: tuple, res: float, sens: str, data_path: str, filename: str
):
    old_span = osa.span
    old_res = osa.resolution
    osa.span = span
    osa.resolution = res
    osa.sensitivity = sens
    res = osa.resolution
    span = osa.span
    sens = osa.sensitivity
    filename = os.path.join(
        data_path,
        filename + f"_span={span[0]:.1f}-{span[1]:.1f}_res={res}_sens={sens}.csv",
    )
    osa.sweeptype = "SGL"
    osa.sweep()
    spectrum = np.vstack((osa.wavelengths, osa.powers))
    with open(filename, "wb") as f:
        np.savetxt(f, spectrum, delimiter=",")
    osa.sweeptype = "RPT"
    osa.span = old_span
    osa.resolution = old_res
    osa.sweep()


data_dir = "./data"
data_child_dir = "tisa_pump_superk_spectra"
data_path = os.path.join(data_dir, data_child_dir)
filename = f"entire-setup_pumps-off_EO1300SPF-after-TMSI_FELH1500_FESH1000"
if not os.path.exists(data_path):
    os.makedirs(data_path)
span = 700, 1700
res = 0.5
sens = "SHI1"
sweep_and_save_spectrum(short_osa, span, res, sens, data_path, filename)
# %%
data_dir_load = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\OD_large_sep_using_superK\data\tisa_pump_superk_spectra"
filename = r"entire-setup-and-1060coupler_superK-port1_tisa-port2_span=700.0-1700.0_res=0.5_sens=SMID.csv"
with open(os.path.join(data_dir_load, filename), "rb") as f:
    spectrum = np.loadtxt(f, delimiter=",")
spectrum[1][spectrum[1] < -90] = np.nan
plt.plot(spectrum[0], spectrum[1])
#
plt.vlines(850, -90, -20, color="k")
plt.vlines(920, -90, -20, color="k")
