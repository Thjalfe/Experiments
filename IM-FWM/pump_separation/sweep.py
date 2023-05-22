import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import traceback
from funcs.experiment import run_experiment, new_sig_start, make_pump_power_equal

# from util_funcs import load_raw_data
sys.path.append("../../../InstrumentControl/InstrumentControl")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from OSA_control import OSA
from laser_control import laser, TiSapphire
from verdi import verdiLaser
from amonicsEDFA import EDFA
from picoscope2000 import PicoScope2000a

plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
GPIB_val = 1
# |%%--%%| <mp5RwPLNWn|ZDg4J4igqj>
TiSa = TiSapphire(3)
ando1_start = 1571.5
ando2_start = 1570.5
pico = PicoScope2000a()
ando_pow_start = 0
ando1 = laser("ando", ando1_start, power=ando_pow_start, GPIB_num=GPIB_val)
ando2 = laser("ando2", ando2_start, power=ando_pow_start, GPIB_num=GPIB_val)
time.sleep(0.1)
ando1.laserON()
ando2.laserON()
verdi = verdiLaser(ioport="COM4")
# edfa = EDFA("COM6")
osa = OSA(
    ando2_start - 2,
    ando1_start + 2,
    resolution=0.05,
    GPIB_num=[GPIB_val, 19],
    sweeptype="SGL",
)
# osa2 = OSA(
#     ando2_start - 2,
#     ando1_start + 2,
#     resolution=0.05,
#     GPIB_num=[GPIB_val, 18],
#     sweeptype="SGL",
# )

# even_ando_power(ando1, ando2, ando_pow_start)
# |%%--%%| <P8MPqQXkmI|vD92Ji6HUF>
wl_tot = 0.8
del_wl = 0.1
num_sweeps = int(wl_tot / del_wl)
mean_pump_wl = 1571
max_pump_sep = 12
ando1_wl = np.arange(mean_pump_wl + 0.5, mean_pump_wl + max_pump_sep / 2, 0.5)
ando2_wl = np.arange(mean_pump_wl - 0.5, mean_pump_wl - max_pump_sep / 2, -0.5)
sig_start = 982
full = np.append(ando1_wl, ando2_wl)
if not len(ando2_wl) == len(ando1_wl):
    raise IndexError(
        f"ando1 and ando2 wl arrays must be the same length, currently they have lengths {len(ando1_wl)} and {len(ando2_wl)} respectively"
    )
EDFA_power = [2.3]
data_folder = []
for pow in EDFA_power:
    data_folder.append(f"./data/CW/{pow}W/")


def start_up(verdiPower, verdi, edfa):
    if verdi is not None:
        verdi.setPower(verdiPower)
        time.sleep(0.1)
        verdi.activeON()
        time.sleep(0.1)
        verdi.setShutter(1)
    if edfa is not None:
        edfa.set_status(1)
        time.sleep(0.1)
        edfa.activeON()


start_up(4.5, verdi, None)
# |%%--%%| <5kdYCanZM7|3B182la2sh>
equal_pump_power = True
log_pm = False
try:
    iter = 0
    while iter < 10:
        for i in range(len(data_folder)):
            try:
                edfa.set_current(int(EDFA_power[i] * 1000))
            except NameError:
                pass
            TiSa.set_wavelength(sig_start, OSA_GPIB_num=[GPIB_val, 19])
            run_experiment(
                data_folder[i],
                ando1,
                ando2,
                TiSa,
                ando1_wl,
                ando2_wl,
                num_sweeps,
                del_wl,
                wl_tot,
                sig_start,
                adjust_laser_wavelengths=True,
                equal_pump_power=equal_pump_power,
                log_pm=log_pm,
                OSA_sens="SHI1",
                OSA_GPIB_num=[GPIB_val, 19],
                sortpeaksby="red",
            )
            iter += 1
except Exception as e:
    print(f"An error occured: {e}")
    print(traceback.format_exc())
finally:
    edfa.activeOFF()
    verdi.shutdown()
    ando1.laserOFF()
    ando2.laserOFF()
# |%%--%%| <jNYLSPMeUe|ub8NueersO>
# redo measurements
data_folder = ["./data/CW/2.3W/", "./data/CW/2.15W/", "./data/CW/2W/"]
pump_sep = np.array([12])
mean_pump_wl = 1571


def redo_specific_pump_sep(pump_sep, mean_pump_wl, data_folder):
    ando1_wl = mean_pump_wl + pump_sep / 2
    ando2_wl = mean_pump_wl - pump_sep / 2
    sig_start = np.zeros(len(pump_sep))
    for i in range(len(pump_sep)):
        sig_start[i] = new_sig_start(
            data_folder, ando2_wl[i], ando1_wl[i], wl_tot, -40, "red"
        )
    return ando1_wl, ando2_wl, sig_start


ando1_wl, ando2_wl, sig_start = redo_specific_pump_sep(
    pump_sep, mean_pump_wl, data_folder[2]
)
# |%%--%%| <laBRSax0MP|ei5Ru2glsb>
# Sweeping pumps and checking their FWM
import os

data_folder = ["./data/pump_DFWM/const_pump_sep", "./data/pump_DFWM/const_mean_pump_wl"]
ando1_array = np.arange(1567, 1581)
ando2_array = np.arange(1565, 1579)
mean_pump_wl = 1571
max_pump_sep = 12
ando1_array = np.arange(mean_pump_wl + 0.5, mean_pump_wl + max_pump_sep / 2, 0.5)
ando2_array = np.arange(mean_pump_wl - 0.5, mean_pump_wl - max_pump_sep / 2, -0.5)
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.5, 1]
equal_pump_power = False
ando1.laserON()
ando2.laserON()
for i in range(len(duty_cycle)):
    folder_name = f"{data_folder[1]}/duty_cycle_{duty_cycle[i]}/"
    os.makedirs(folder_name, exist_ok=True)
    pico.awg.set_square_wave(pulse_freq, duty_cycle[i])
    for idx_wl, (wl1, wl2) in enumerate(zip(ando1_array, ando2_array)):
        file_name = f"{folder_name}/{wl1}_{wl2}"
        ando1.set_wavelength(wl1)
        ando2.set_wavelength(wl2)
        time.sleep(0.1)
        ando1.adjust_power()
        ando2.adjust_power()
        time.sleep(0.1)
        if equal_pump_power:
            make_pump_power_equal(ando1, ando2, wl1, wl2, OSA_GPIB_num=[GPIB_val, 19])
        wl_diff = np.abs(wl1 - wl2)
        lower_lim = np.min([wl1, wl2]) - wl_diff * 2.5
        upper_lim = np.max([wl1, wl2]) + wl_diff * 2.5
        osa.set_span(lower_lim, upper_lim)
        osa.sweep()
        osa.save(file_name)
