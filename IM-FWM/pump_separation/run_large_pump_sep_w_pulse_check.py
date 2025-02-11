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
from picoscope2000 import PicoScope2000a
from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons
from arduino_pm import ArduinoADC
from tektronix_oscilloscope import TektronixOscilloscope
from ipg_edfa import IPGEDFA


plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
ando1_start = 1608
ando2_start = 1603
duty_cycle = [0.1, 0.2, 0.5, 1]
pulse_freq = 1 * 10**5
# scope = TektronixOscilloscope()
# edfa1 = EDFA("COM12")
arduino = ArduinoADC("COM11")
ipg_edfa = IPGEDFA(connection_mode="GPIB", GPIB_address=17)
pol_con1 = ThorlabsMPC320(serial_no_idx=0)
pol_con2 = ThorlabsMPC320(serial_no_idx=1)
# edfa2 = EDFA("COM5")
pico = PicoScope2000a()
pico.awg.set_square_wave_duty_cycle(pulse_freq, duty_cycle[1])
ando_pow_start = 0
ando1 = AndoLaser(ando1_start, GPIB_address=23, power=-8)
ando2 = AndoLaser(ando2_start, GPIB_address=24, power=10)
time.sleep(0.1)
ando1.laser_on()
ando2.laser_on()
# TiSa = TiSapphire(3)
# verdi = VerdiLaser(com_port=4)
osa = OSA(
    970,
    979,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="SGL",
)
EDFA_power = [2300]
data_folder = []
for pow in EDFA_power:
    data_folder.append(f"./data/CW/{pow}W/")


def start_edfa(edfa, edfa_pow, mode="APC"):
    if edfa is not None:
        edfa.set_mode(mode)
        time.sleep(0.1)
        edfa.set_status(1)
        time.sleep(0.1)
        edfa.set_current(edfa_pow)
        time.sleep(0.1)
        edfa.active_on()


def start_up(verdiPower, verdi, ando1, ando2):
    if verdi is not None:
        verdi.set_power(verdiPower)
        time.sleep(0.1)
        verdi.active_on()
        time.sleep(0.1)
        verdi.set_shutter(1)
    ando1.laser_on()
    ando2.laser_on()


# start_edfa(edfa1, EDFA_power[0])
# start_edfa(edfa2, 1150)
start_up(9, verdi, ando1, ando2)
# %%

optimize_multiple_pol_cons(
    arduino, pol_con1, pol_con2, max_or_min="max", interval=0.01, tolerance=0.5
)
# %%
print(1 / (1 / ando2.get_wavelength() - 1 / ando1.get_wavelength() + 1 / 1 / 964.61))
# %%
pulse_freq = 10 * 10**4
dc = 1
pico.awg.set_square_pulse_arb(pulse_freq, 0.0, dc)
# pulse_freq = 10**5
# pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.5)
# osa.sweep()
# %%
osa_start_lims = [950, 980]
ando1_start_wl = 1573
ando1_delta_wl = 30
ando1_step_size = 5
tisa_tot_delta_wl = 15
tisa_step_size = 1
data_dict = {}
for i in range(ando1_delta_wl // ando1_step_size):
    t0 = time.time()
    ando1.set_wavelength(ando1_start_wl + i * ando1_step_size)
    data_dict[ando1.get_wavelength()] = {}
    for j in range(tisa_tot_delta_wl // tisa_step_size):
        osa.set_span(
            osa_start_lims[0] + j * tisa_step_size,
            osa_start_lims[1] + j * tisa_step_size,
        )
        osa.sweep()
        data_dict[ando1.get_wavelength()][TiSa.get_wavelength()] = np.array(
            [osa.wavelengths, osa.powers]
        )
        TiSa.delta_wl_nm(tisa_step_size)
        print(f"Done with {TiSa.get_wavelength()}")
    t1 = time.time()
    TiSa.delta_wl_nm(-tisa_tot_delta_wl)
    print(f"Done with {ando1.get_wavelength()} in {t1-t0:.2f} s")
# %%
pump_wl = ando1.get_wavelength()
s_wl = 1 / (-1 / 969 + 1 / 975.46 + 1 / pump_wl)
print(f"s_wl: {s_wl:.2f} nm, separation: {s_wl-pump_wl:.2f} nm")
# %%
osa.get_spectrum()
wl = osa.wavelengths
power = osa.powers
data_short[dc] = np.array([wl, power])
# %%
# data_dict = {}
data_dict["short_wl"] = data_short
# %%
with open(
    r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\ok_distant_bs.pkl",
    "wb",
) as f:
    pickle.dump(data_dict, f)
# %%
pulse_freq = 0.2 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5]
# duty_cycle = [0.1]
time_ax = []
voltage = []
for dc in duty_cycle:
    time.sleep(2)
    pico.awg.set_square_wave(pulse_freq, dc)
    time.sleep(2)
    time_, voltage_ = scope.save_waveform(1)
    time_ax.append(time_)
    voltage.append(voltage_)
# %%
import pickle

traces = {"time": time_ax, "voltage": voltage, "duty_cycle": duty_cycle}
with open(
    f"./data/scope_traces/sig_idler_traces/sig_traces/{pulse_freq*10**-3}kHz_pump_sep15nm.pkl",
    "wb",
) as f:
    pickle.dump(traces, f)
# %%
idx = 0
fig, ax = plt.subplots()
plt.plot(time_ax[idx], voltage[idx])
# %%
ando1_wl_start = 1608
ando2_wl_start = ando1_wl_start - 2
step_size = 1
num_steps = 30
data_dict = {}
for i in range(num_steps):
    ando1_wl = ando1_wl_start - i * step_size
    ando1.set_wavelength(ando1_wl)
    ando2_wl = ando2_wl_start - i * step_size
    ando2.set_wavelength(ando2_wl)
    time.sleep(0.5)
    osa.sweep()
    wl = osa.wavelengths
    power = osa.powers
    data_dict[ando1_wl] = np.array([wl, power])
    print(f"Done with {ando1_wl} nm")
# %%
ando1_wl_array = np.arange(1588, 1599, 2)
ando1_wl_array = np.array([1608])
wl_tot = 35
step_size = 0.1
num_steps = int(wl_tot / step_size)
data_dict = {}
for ando1_wl in ando1_wl_array:
    ando1.set_wavelength(ando1_wl)
    wl_start = ando1_wl - 40
    data_dict[ando1_wl] = {}
    for j in range(num_steps):
        wl_ando = wl_start + j * step_size
        ando2.set_wavelength(wl_ando)
        time.sleep(0.1)
        osa.sweep()
        wl = osa.wavelengths
        power = osa.powers
        data_dict[ando1_wl][wl_ando] = np.array([wl, power])
    print(f"Done with {ando1_wl} nm")

with open(
    r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\ando1_2_sweep_965TiSa.pkl",
    "wb",
) as f:
    pickle.dump(data_dict, f)
# %%
power[power < -80] = np.nan
max_pow_loc = np.nanargmax(power)
max_wl = wl[max_pow_loc]
p_wl = 1603
s_wl = 1608
i_wl = 1 / (1 / s_wl - 1 / p_wl + 1 / max_wl)
fig, ax = plt.subplots()
ax.plot(wl, power)
ax.axvline(i_wl, color="k")
# %%
with open("./data/ando1_2_sweep_965TiSa.pkl", "rb") as f:
    data_dict = pickle.load(f)
data = data_dict[1608]
p2_wl = np.array(list(data.keys()))
fig, ax = plt.subplots()
for p2 in p2_wl:
    ax.plot(data[p2][0], data[p2][1], label=f"{p2} nm")
# plt.legend()
plt.show()
# %%
mean_pump_wl = 1588
ando1_wl_lst = np.append(1589, np.arange(mean_pump_wl + 2.5, 1608 + 2.5, 2.5))
ando2_wl_lst = np.append(1587, np.arange(mean_pump_wl - 2.5, 1568 - 2.5, -2.5))
pump_sep_idx = 6
ando1_wl = ando1_wl_lst[pump_sep_idx]
ando1.set_wavelength(ando1_wl)
ando2_wl = ando2_wl_lst[pump_sep_idx]
ando2.set_wavelength(ando2_wl)
# |%%--%%| <g8zUm8hN1t|sIeoGAOst5>
# Check different duty cycle CE experiment
import pickle

data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/large_pump_sep_within_L_band/pol_check/minimized"
file_name = f"pumps_{ando2_wl}_{ando1_wl}"
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]
osa.set_res(0.05)
step_size = 0.1
num_steps = 1
# osa.set_sens("SHI1")
osa.set_sweeptype("SGL")
osa.sweep()
num_for_mean = 3
osa_data = {
    "pulse_freq": pulse_freq,
    "duty_cycle": duty_cycle,
    "wavelengths": np.zeros(
        (len(duty_cycle), num_steps, num_for_mean, len(osa.wavelengths))
    ),
    "powers": np.zeros((len(duty_cycle), num_steps, num_for_mean, len(osa.powers))),
}
t0 = time.time()
for k in range(num_steps):
    if k > 0:
        TiSa.delta_wl_nm(step_size)
    for i in range(len(duty_cycle)):
        print("Duty cycle: ", duty_cycle[i])
        for j in range(num_for_mean):
            pico.awg.set_square_wave(pulse_freq, duty_cycle[i])
            osa.sweep()
            osa_data["wavelengths"][i, k, j, :] = osa.wavelengths
            osa_data["powers"][i, k, j, :] = osa.powers
t1 = time.time()
print(f"Time taken: {t1-t0:.2f} s")
# with open(f"{data_folder}/{file_name}_oscilloscope.pkl", "wb") as f:
#     pickle.dump(oscilloscope_data, f)
with open(f"{data_folder}/{file_name}_spectra.pkl", "wb") as f:
    pickle.dump(osa_data, f)
# |%%--%%| <BURpmFcmOb|Jnpn8nizq6>
# Stability of idler measurements
p1_wl = ando1.get_wavelength()
p2_wl = ando2.get_wavelength()
osa.set_res(0.01)
pico.awg.set_square_wave(pulse_freq, 0.5)
save_spec_dir = f"C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/C_plus_L_band/idler_stability_meas/after_mode_imaging/{p1_wl:.2f}nm_{p2_wl:.2f}nm"
if not os.path.exists(save_spec_dir):
    os.makedirs(save_spec_dir)
for i in range(50):
    osa.sweep()
    osa.save(f"{save_spec_dir}/sig_idler_{i}")
    print(i)
# |%%--%%| <Jnpn8nizq6|p0nkaUgozM>
ando1.set_wavelength(1608)
ando2.set_wavelength(1532)
# |%%--%%| <p0nkaUgozM|BURpmFcmOb>
p1_wl = ando1.get_wavelength()
p2_wl = ando2.get_wavelength()
osa.set_res(0.1)
pico.awg.set_square_wave(pulse_freq, 0.5)
save_spec_dir = f"C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/C_plus_L_band/after_pump_sep/5us_50percent/{p1_wl:.2f}nm_{p2_wl:.2f}nm"
if not os.path.exists(save_spec_dir):
    os.makedirs(save_spec_dir)
iter = 0
while True:
    TiSa.delta_wl_nm(0.1)
    osa.sweep()
    osa.save(f"{save_spec_dir}/spec_{iter}")
    iter += 1
# %%
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))
