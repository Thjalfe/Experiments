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

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump2_start = 1551
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.2)
ando_pow_start = 0
pump2_laser = AndoLaser(pump2_start, GPIB_address=24, power=8)
time.sleep(0.1)
pump2_laser.enable()
tisa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    979,
    990,
    resolution=0.1,
    GPIB_address=19,
    sweeptype="SGL",
    trace="a",
)

verdi.power = 6
verdi.shutter = 0
time.sleep(0.5)
verdi.shutter = 1


# %%
def expected_idler_wl(p2_wl, sig_wl, p1_wl=1571):
    return 1 / (-1 / p2_wl + 1 / sig_wl + 1 / p1_wl)


def mark_sig_idler(osa: OSA, p2_wl: float, p1_wl: float = 1577):
    def find_p1_phasematched_wl(p2_wl: float, sig_wl: float, idler_wl: float):
        return 1 / (1 / p2_wl - 1 / sig_wl + 1 / idler_wl)

    # osa.update_spectrum()
    tisa_wl = osa.wavelengths[np.argmax(osa.powers)]
    idler_wl = expected_idler_wl(p2_wl, tisa_wl, p1_wl)
    tisa_power = np.max(osa.powers)
    idler_region_idxs = np.where(
        np.logical_and(osa.wavelengths > idler_wl - 1, osa.wavelengths < idler_wl + 1)
    )
    idler_power = np.max(osa.powers[idler_region_idxs])
    real_idler_wl = osa.wavelengths[idler_region_idxs][
        np.argmax(osa.powers[idler_region_idxs])
    ]
    osa.set_wavelength_marker(1, tisa_wl)
    osa.set_wavelength_marker(2, real_idler_wl)
    osa.set_power_marker(3, tisa_power)
    osa.set_power_marker(4, idler_power)
    modelocked_phasematched_wl = find_p1_phasematched_wl(p2_wl, tisa_wl, real_idler_wl)
    print(f"Modelocked phasematched wavelength: {modelocked_phasematched_wl:.2f} nm")
    print(f"Current CE average: {idler_power -tisa_power:.2f} dBm")
    print(f"Current CE peak estimate: {(idler_power -tisa_power+ 50):.2f} dBm")
    return modelocked_phasematched_wl, idler_wl, tisa_wl


p1_pm_wl, idler_wl, sig_wl = mark_sig_idler(osa, pump2_laser.wavelength)
# tisa_wl = 983.4
# idler_wl = expected_idler_wl(pump2_start, tisa_wl)
# osa.span = (sig_wl - 1, idler_wl + 2)
# osa.set_wavelength_marker(1, tisa_wl)
# osa.set_wavelength_marker(2, idler_wl)
# %% fast scan
osa.stop_sweep()
osa.sweeptype = "SGL"
for i in range(10):
    tisa.delta_wl_nm(0.25)
    osa.sweep()
# %%  linear fit
fit_file = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\processing\fits\modelocked_1571_pump_fit.txt"
fit = np.loadtxt(fit_file)
fit_fn = np.poly1d(fit)

# %%
data_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\modelocked_1571_pump\modelocked_as_pump\p2_wl=1563"
osa.stop_sweep()
osa.sweeptype = "SGL"
p1_wl = 1578
for i in range(60):
    tisa.delta_wl_nm(0.1)
    osa.sweep()
    p1_pm_wl, idler_wl, sig_wl = mark_sig_idler(
        osa, pump2_laser.wavelength, p1_wl=p1_wl
    )
    p1_wl = p1_pm_wl

    spectrum = np.vstack((osa.wavelengths, osa.powers))
    np.savetxt(
        os.path.join(
            data_dir,
            f"spec_p2wl={pump2_laser.wavelength:.1f}_p1pm_wl={p1_wl:.2f}_iter={i}.csv",
        ),
        spectrum,
        fmt="%f",
        delimiter=",",
    )
