import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import traceback
from funcs.experiment import (
    run_tisa_sweep_all_pump_wls,
    new_sig_start,
    make_pump_power_equal,
    run_tisa_sweep_single_pump_wl,
)

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
GPIB_val = 0
# |%%--%%| <mp5RwPLNWn|EVutLKJ1oz>
TiSa = TiSapphire(3)
ando1_start = 1589
ando2_start = 1587
duty_cycle = [0.1, 0.5, 1]
pulse_freq = 1 * 10**5
pico = PicoScope2000a()
pico.awg.set_square_wave(pulse_freq, duty_cycle[-1])
ando_pow_start = 0
ando1 = laser("ando", ando1_start, power=ando_pow_start, GPIB_num=GPIB_val)
ando2 = laser("ando2", ando2_start, power=ando_pow_start, GPIB_num=GPIB_val)
time.sleep(0.1)
ando1.laserON()
ando2.laserON()
verdi = verdiLaser(ioport="COM4")
edfa = EDFA("COM7")
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
# |%%--%%| <BLQ3WeoBDX|4bRQN9CPDT>
wl_tot = 0.72
del_wl = 0.08
num_sweeps = int(wl_tot / del_wl)
mean_pump_wl = 1588
ando1_wl_lst = np.append(1589, np.arange(mean_pump_wl + 2.5, 1608 + 2.5, 2.5))
ando2_wl_lst = np.append(1587, np.arange(mean_pump_wl - 2.5, 1568 - 2.5, -2.5))
#|%%--%%| <4bRQN9CPDT|iv3c5bes4N>
# full = np.append(ando1_wl, ando2_wl)
# if not len(ando2_wl) == len(ando1_wl):
#     raise IndexError(
#         f"ando1 and ando2 wl arrays must be the same length, currently they have lengths {len(ando1_wl)} and {len(ando2_wl)} respectively"
#     )
EDFA_power = [2.3]
data_folder = []
for pow in EDFA_power:
    data_folder.append(f"./data/CW/{pow}W/")


def start_up(verdiPower, verdi, edfa, edfa_pow, ando1, ando2):
    if verdi is not None:
        verdi.setPower(verdiPower)
        time.sleep(0.1)
        verdi.activeON()
        time.sleep(0.1)
        verdi.setShutter(1)
    if edfa is not None:
        edfa.set_status(1)
        time.sleep(0.1)
        edfa.set_current(edfa_pow)
        time.sleep(0.1)
        edfa.activeON()
    ando1.laserON()
    ando2.laserON()


start_up(7, verdi, edfa, 2300, ando1, ando2)
#|%%--%%| <awDAJLovld|QM8VeyRH0i>
num = -9
ando1_wl = ando1_wl_lst[num]
ando1.set_wavelength(ando1_wl)
ando2_wl = ando2_wl_lst[num]
ando2.set_wavelength(ando2_wl)
# |%%--%%| <iv3c5bes4N|T6DbNJxS1g>
# Check different duty cycle CE experiment
import pickle

data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/large_pump_sep_within_L_band/"
file_name = f"pumps_{ando2_wl}_{ando1_wl}"
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]
osa.set_res(0.1)
osa.sweep()
oscilloscope_data = {
    "pulse_freq": pulse_freq,
    "duty_cycle": duty_cycle,
    "time_ref": np.zeros((len(duty_cycle), 1250)),
    "voltage_ref": np.zeros((len(duty_cycle), 1250)),
    "time_sig": np.zeros((len(duty_cycle), 1250)),
    "voltage_sig": np.zeros((len(duty_cycle), 1250)),
}
osa.get_spectrum()
osa_data = {
    "pulse_freq": pulse_freq,
    "duty_cycle": duty_cycle,
    "wavelengths": np.zeros((len(duty_cycle), 5, len(osa.wavelengths))),
    "powers": np.zeros((len(duty_cycle), 5, len(osa.powers))),
}
t0 = time.time()
for i in range(len(duty_cycle)):
    for j in range(5):
        pico.awg.set_square_wave(pulse_freq, duty_cycle[i])
        pico.oscilloscope.set_params(
            channel=2,
            enabled=0,
        )
        pico.oscilloscope.set_params(
            channel=0,
            enabled=1,
            channel_range=7,
            oversample=2,
            timebase=3,
            threshold=50,
        )
        pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
        (
            time_axis_ref,
            y_ref,
            overflow,
            adc1,
            pulse_length,
            pulse_dist,
        ) = pico.oscilloscope.run_scope(threshold_high=0.5, threshold_low=0.5)
        pico.oscilloscope.set_params(
            channel=0,
            enabled=0,
        )
        range_list = [7, 7, 7, 5, 5, 5]
        oscilloscope_range = range_list[i]
        pico.oscilloscope.set_params(
            channel=2,
            enabled=1,
            channel_range=7,
            oversample=2,
            timebase=3,
            threshold=50,
        )
        pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
        (
            time_axis,
            y,
            overflow,
            adc,
            pulse_length_out,
            pulse_dist_out,
        ) = pico.oscilloscope.run_scope()
        oscilloscope_data["time_ref"][i, :] = time_axis_ref
        oscilloscope_data["voltage_ref"][i, :] = y_ref
        oscilloscope_data["time_sig"][i, :] = time_axis
        oscilloscope_data["voltage_sig"][i, :] = y
        osa.sweep()
        osa_data["wavelengths"][i, j, :] = osa.wavelengths
        osa_data["powers"][i, j, :] = osa.powers
t1 = time.time()
print(f"Time taken: {t1-t0:.2f} s")
with open(f"{data_folder}/{file_name}_oscilloscope.pkl", "wb") as f:
    pickle.dump(oscilloscope_data, f)
with open(f"{data_folder}/{file_name}_spectra.pkl", "wb") as f:
    pickle.dump(osa_data, f)
