# %%
import numpy as np
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
from amonics_edfa import EDFA
from thorlabs_mpc320 import ThorlabsMPC320, optimize_multiple_pol_cons
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA


plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
ando_start = 1550
arduino = ArduinoADC("COM11")
ipg_edfa = IPGEDFA(connection_mode="GPIB", GPIB_address=17)
pol_con = ThorlabsMPC320(serial_no_idx=1)
ando_pow_start = 0
ando_laser = AndoLaser(ando_start, GPIB_address=24, power=8)
time.sleep(0.1)
ando_laser.laser_on()
TiSa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    1545,
    1555,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="RPT",
)

verdi.set_power(5)
verdi.set_shutter(1)


# %%
def idler_wl(tisa_wl, c_band_wl, pulsed_wl=1064):
    return 1 / (-1 / tisa_wl + 1 / c_band_wl + 1 / pulsed_wl)


# %%
ando_wl_start = 1535
ando_wl_end = 1565
ando_wl_step = 0.25
ando_wl_range = np.arange(ando_wl_start, ando_wl_end + ando_wl_step, ando_wl_step)
spectra = {}
osa.set_sens("SHI2")
osa.set_res(0.1)
osa.set_sweeptype("SGL")
for wl in ando_wl_range:
    ando_laser.set_wavelength(wl)
    expected_idler_wl = idler_wl(998, wl)
    osa_span = (expected_idler_wl - 4, expected_idler_wl + 4)
    osa.set_span(*osa_span)
    osa.sweep()
    wavelength = osa.wavelengths
    power = osa.powers
    spectra[wl] = np.array([wavelength, power])
    spectra[wl][spectra[wl] < -100] = np.nan
# %%
fig, ax = plt.subplots()
for wl, spectrum in spectra.items():
    spectrum[spectrum < -80] = np.nan
    # norm_power = spectrum[1] - np.nanmax(spectrum[1])
    norm_power = spectrum[1]
    ax.plot(spectrum[0], norm_power, label=wl)
# ax.legend()
# %%
wl = 1549.0
fig, ax = plt.subplots()
ax.plot(spectra[wl][0], spectra[wl][1])
