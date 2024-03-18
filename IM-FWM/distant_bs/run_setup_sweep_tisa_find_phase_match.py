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
pump1_start = 1580
seed_start = 1600
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
seed_laser = PhotoneticsLaser(seed_start, GPIB_address=10, power=6)
# pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
# pump2_laser = PhotoneticsLaser(pump2_start, GPIB_address=10, power=8)
time.sleep(0.1)
pump1_laser.enable()
seed_laser.enable()
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
def idler_wl_expected(sig_wl: float, pump1_wl: float, pump2_wl: float = 1064) -> float:
    return 1 / (1 / sig_wl - 1 / pump2_wl + 1 / pump1_wl)


pulsed_pump = 1029.38
pump_name = "toptica"
base_path = os.getcwd()
data_path = os.path.join(base_path, f"data\\wide_tisa_sweeps\\toptica")
if not os.path.exists(data_path):
    os.makedirs(data_path)

pump1_start = 1605
pump1_laser.wavelength = pump1_start
tisa_stepsize = -0.1
tisa_start_wl = 992.5
tisa_tot_mov = -50
tisa_wl = tisa_start_wl
num_steps = int(np.abs(tisa_tot_mov / tisa_stepsize))
# %%
osa.stop_sweep()
osa.sweeptype = "SGL"
osa.resolution = 0.1
osa.sensitivity = "SHI1"
data_dict = {
    "params": {
        "osa_res": osa.resolution,
        "toptica_3db_bw_in": "1.5nm",
        "toptica_pin_before_fiber": "46mW",
    }
}
t0 = time.time()
for i in range(num_steps):
    idler_wl = idler_wl_expected(tisa_wl, pump1_start, pump2_wl=pulsed_pump)
    osa.span = (idler_wl - 7, idler_wl + 7)
    osa.sweep()
    wavelengths = osa.wavelengths
    powers = osa.powers
    data_dict[tisa_wl] = np.vstack((wavelengths, powers))
    tisa.delta_wl_nm(tisa_stepsize)
    tisa_wl += tisa_stepsize
    print(f"Step {i+1}/{num_steps} done")
    print(f"Current TiSa wavelength: {tisa_wl}")
    print(f"Time elapsed: {time.time()-t0:.2f} s")
with open(
    os.path.join(
        data_path,
        f"tisa_sweep_pump1wl={pump1_start}_pump2wl={pulsed_pump:.1f}_50%toptica.pkl",
    ),
    "wb",
) as f:
    pickle.dump(data_dict, f)


# %%
def get_full_spectrum(
    osa: OSA, shortest_wl: float, longest_wl: float, res: float = 2, sens: str = "SMID"
) -> np.ndarray:
    osa.stop_sweep()
    osa.sweeptype = "SGL"
    osa.resolution = res
    osa.sensitivity = sens
    osa.span = (shortest_wl, longest_wl)
    osa.sweep()
    wavelengths = osa.wavelengths
    powers = osa.powers
    return np.vstack((wavelengths, powers))


full_spec = get_full_spectrum(osa, 985, 1620)
# save as csv
# %%
np.savetxt(
    os.path.join(
        data_path, f"full_spec_pump1wl={pump1_start}_pump2wl={pulsed_pump:.1f}.csv"
    ),
    full_spec,
    delimiter=",",
)
