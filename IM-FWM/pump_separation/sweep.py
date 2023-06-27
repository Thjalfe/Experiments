import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import traceback
from funcs.experiment import run_tisa_sweep_all_pump_wls, new_sig_start, make_pump_power_equal, run_tisa_sweep_single_pump_wl

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
# |%%--%%| <BLQ3WeoBDX|vD92Ji6HUF>
wl_tot = 0.72
del_wl = 0.08
num_sweeps = int(wl_tot / del_wl)
mean_pump_wl = 1588
max_pump_sep = 12
ando_wls = np.loadtxt("./data/calibration/ando_wl_calibration/1565.5-1576.5nm_0.5step.csv")
ando1_wl = ando_wls[0]
ando2_wl = ando_wls[1]
ando1_wl_names = np.arange(mean_pump_wl + 0.5, mean_pump_wl + max_pump_sep / 2, 0.5)
ando2_wl_names = np.arange(mean_pump_wl - 0.5, mean_pump_wl - max_pump_sep / 2, -0.5)
sig_start = 982.1
# full = np.append(ando1_wl, ando2_wl)
# if not len(ando2_wl) == len(ando1_wl):
#     raise IndexError(
#         f"ando1 and ando2 wl arrays must be the same length, currently they have lengths {len(ando1_wl)} and {len(ando2_wl)} respectively"
#     )
EDFA_power = [2.3]
data_folder = []
for pow in EDFA_power:
    data_folder.append(f"./data/CW/{pow}W/")


def start_up(verdiPower, verdi, edfa, ando1, ando2):
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
    # ando1.laserON()
    # ando2.laserON()


start_up(7, verdi, edfa, ando1, ando2)
#|%%--%%| <DOYbnxfhwB|KH1h6SDYCC>
# Sweep pump wavelengths automatically
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
            run_tisa_sweep_all_pump_wls(
                data_folder[i],
                ando1,
                ando2,
                TiSa,
                ando1_wl,
                ando2_wl,
                ando1_wl_names,
                ando2_wl_names,
                num_sweeps,
                del_wl,
                wl_tot,
                sig_start,
                adjust_laser_wavelengths=False,
                equal_pump_power=equal_pump_power,
                log_pm=log_pm,
                OSA_sens="SMID",
                OSA_GPIB_num=[GPIB_val, 19],
                sortpeaksby="red",
                sweep_pumps=False,
            )
            iter += 1
except Exception as e:
    print(f"An error occured: {e}")
    print(traceback.format_exc())
finally:
    try:
        edfa.activeOFF()
    except NameError:
        pass
    verdi.shutdown()
    ando1.laserOFF()
    ando2.laserOFF()
#|%%--%%| <KH1h6SDYCC|DLNO44xNAK>
# same but not sweeping pump wls
data_folder = ["./data/CW/2.3W/pol_dependence/Optimized_pol_for_all_pump_seps/const_sigstart_every_pumpsep/"]
data_folder = ["C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/40nm_sep/10us_window_0.5duty/"]
# ando_wl_idx = 0
sig_start = [982.1, 982, 981.7, 981.5, 981.3, 981.2, 980.9, 980.6, 980.4, 980.2, 979.9]
sig_start = [970]
equal_pump_power = True
ando1_wl = [1608]
ando2_wl = [1568]
ando1_wl_names = [1608]
ando2_wl_names = [1568]
log_pm = False
num_iters = 1
ando_wl_idx = 0
wl_tot = 5
del_wl = 0.1
num_sweeps = int(wl_tot / del_wl)
#|%%--%%| <DLNO44xNAK|ZgE8wah9Xb>
try:
    iter = 0
    while iter < num_iters:
        # rand_start_offset = np.round(np.random.uniform(-0.1, 0.1), 3)
        rand_start_offset = 0
        TiSa.set_wavelength(sig_start[ando_wl_idx] + rand_start_offset, OSA_GPIB_num=[GPIB_val, 19])
        # try:
        #     edfa.set_current(int(EDFA_power * 1000))
        # except NameError:
        #     pass
        run_tisa_sweep_single_pump_wl(
            # f"{data_folder[0]}{iter + 1}",
            f"{data_folder[0]}",
            ando1,
            ando2,
            TiSa,
            ando1_wl[ando_wl_idx],
            ando2_wl[ando_wl_idx],
            ando1_wl_names[ando_wl_idx],
            ando2_wl_names[ando_wl_idx],
            num_sweeps,
            del_wl,
            wl_tot,
            sig_start[ando_wl_idx],
            adjust_laser_wavelengths=False,
            equal_pump_power=equal_pump_power,
            log_pm=log_pm,
            OSA_sens="SHI1",
            OSA_GPIB_num=[GPIB_val, 19],
            sortpeaksby="red",
            make_new_folder_iter=False,
            sig_start_external=True,
        )
        print(f"Finished iteration {iter + 1} of {num_iters}")
        iter += 1
except Exception as e:
    print(f"An error occured: {e}")
    print(traceback.format_exc())
# t1 = time.time()
# print(f"Total time elapsed: {t1 - t0} s")
# finally:
#     try:
#         edfa.activeOFF()
#     except NameError:
#         pass
#     verdi.shutdown()
#     ando1.laserOFF()
#     ando2.laserOFF()
# |%%--%%| <ZgE8wah9Xb|oIj1oTHvpt>
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
C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/40nm_sep/10us_window_0.5duty
#|%%--%%| <oIj1oTHvpt|AO3ennIf7k>
temp_save_dir = "./data/CW/2.3W/pol_dependence/"
wl_idx = 9
ando1.set_wavelength(ando1_wl[wl_idx])
ando2.set_wavelength(ando2_wl[wl_idx])
wl_diff = ando1_wl[wl_idx] - ando2_wl[wl_idx]
file_name = f"pol_dep_optimized_for_{1.00}nm_with_pump_sep_{wl_diff:.2f}nm_wl_optimized_for_10nmpolopt"
#|%%--%%| <p4HJOPndFW|bVuusTei57>
osa.get_spectrum()
osa.save(f"{temp_save_dir}{file_name}")
#|%%--%%| <bVuusTei57|XI2YYf3Cj6>
import glob
files = glob.glob(temp_save_dir + "*.csv")
x = []
y = []
for file in files:
    data = np.loadtxt(file, delimiter=",")
    x.append(data[:, 0])
    y.append(data[:, 1])
for i in range(len(x)):
    plt.plot(x[i], y[i], label=f"{i}")
plt.legend()
plt.show()
# |%%--%%| <XI2YYf3Cj6|6xMrJRx41N>
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
#|%%--%%| <FLCH356Rqy|mpmeGsr3ba>
start_wl1 = ando1_wl[wl_idx]
start_wl2 = ando2_wl[wl_idx]
ando1.set_wavelength(start_wl1)
ando2.set_wavelength(start_wl2)
print(start_wl1)
print(start_wl2)
def temp_change_ando_wl(ando, del_wl):
    ando.set_wavelength(ando.get_wavelength() + del_wl)
    ando.adjust_power()
tot_wl_move = 0.5
step_size = 0.01
size = int(tot_wl_move / step_size)
ando1.set_wavelength(start_wl1 - tot_wl_move / 2)
ando2.set_wavelength(start_wl2 - tot_wl_move / 2)
wls = np.zeros((2, size))
spectra = np.zeros((2, size, len(osa.powers)))
for i in range(size):
    temp_change_ando_wl(ando1, step_size)
    temp_change_ando_wl(ando2, step_size)
    time.sleep(5)
    wls[0, i] = ando1.get_wavelength()
    wls[1, i] = ando2.get_wavelength()
    osa.sweep()
    spectra[0, i, :] = osa.wavelengths
    spectra[1, i, :] = osa.powers
#|%%--%%| <mpmeGsr3ba|UIuH9nX0UQ>
np.save('spectra', spectra, allow_pickle=True)
#|%%--%%| <UIuH9nX0UQ|5S3ztWB3us>
# Check different duty cycle CE experiment
import pickle
data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/pulse/10us_cycle/"
file_name = "pumps_1568_1578"
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]

osa.sweep()
oscilloscope_data = {"pulse_freq": pulse_freq, "duty_cycle": duty_cycle, "time_ref": np.zeros((len(duty_cycle), 1250)), "voltage_ref": np.zeros((len(duty_cycle), 1250)), "time_sig": np.zeros((len(duty_cycle), 1250)), "voltage_sig": np.zeros((len(duty_cycle), 1250))}
osa.get_spectrum()
osa_data = {"pulse_freq": pulse_freq, "duty_cycle": duty_cycle, "wavelengths": np.zeros((len(duty_cycle), len(osa.wavelengths))), "powers": np.zeros((len(duty_cycle), len(osa.powers)))}
for i in range(len(duty_cycle)):
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
    time_axis_ref, y_ref, overflow, adc1, pulse_length, pulse_dist = pico.oscilloscope.run_scope(threshold_high=0.5, threshold_low=0.5)
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
    time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = pico.oscilloscope.run_scope()
    oscilloscope_data["time_ref"][i, :] = time_axis_ref
    oscilloscope_data["voltage_ref"][i, :] = y_ref
    oscilloscope_data["time_sig"][i, :] = time_axis
    oscilloscope_data["voltage_sig"][i, :] = y
    osa.sweep()
    osa_data["wavelengths"][i, :] = osa.wavelengths
    osa_data["powers"][i, :] = osa.powers
with(open(f"{data_folder}/{file_name}_oscilloscope.pkl", "wb")) as f:
    pickle.dump(oscilloscope_data, f)
with open(f"{data_folder}/{file_name}_spectra.pkl", "wb") as f:
    pickle.dump(osa_data, f)
#|%%--%%| <jYQYF4O7ut|J41CEAjSMp>
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]
pico.awg.set_square_wave(pulse_freq, duty_cycle[-2])
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
time_axis1, y1, overflow, adc1, pulse_length, pulse_dist = pico.oscilloscope.run_scope(threshold_high=0.5, threshold_low=0.5)
pico.oscilloscope.set_params(
    channel=0,
    enabled=0,
)
# [7, 7, 7, 5, 5, 5]
pico.oscilloscope.set_params(
    channel=2,
    enabled=1,
    channel_range=7,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = pico.oscilloscope.run_scope()
# fig, ax = plt.subplots(1, 1)
# ax.plot(time_axis1, y1)
# ax1 = ax.twinx()
# ax1.plot(time_axis, y, color='red')
# plt.show()
#|%%--%%| <aSxLF4tLrQ|qAdbd72Hvw>
pico.oscilloscope.set_params(
    channel=3,
    enabled=1,
    channel_range=2,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico.oscilloscope.set_sampling_rate(20000, peaks_wanted=1)
while True:
    time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = pico.oscilloscope.run_scope()
    print(np.mean(y))
    time.sleep(0.1)
# fig, ax = plt.subplots(1, 1)
# ax.plot(time_axis, y)
# plt.show()
#|%%--%%| <qAdbd72Hvw|G4iR0kz8Fb>

import serial.tools.list_ports
ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))
