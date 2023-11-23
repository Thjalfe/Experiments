import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys

from picosdk.functions import PicoSDKCtypesError

sys.path.append(
    r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\InstrumentControl"
)
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser
from amonics_edfa import EDFA
from picoscope2000 import PicoScope2000a

from tektronix_oscilloscope import TektronixOscilloscope

plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
GPIB_val = 0
# |%%--%%| <mp5RwPLNWn|8x5CANACEx>
ando1_start = 1603
ando2_start = 1573
edfa1 = EDFA("COM7")
pico = PicoScope2000a()
pulse_freq = 1 * 10**5
pico.awg.set_square_wave(pulse_freq, 0.5)
ando_pow_start = 0
ando1 = laser("ando", ando1_start, power=ando_pow_start, GPIB_num=GPIB_val)
ando2 = laser("ando2", ando2_start, power=ando_pow_start, GPIB_num=GPIB_val)
time.sleep(0.1)
ando1.laserON()
ando2.laserON()
# osa = OSA(
#     ando2_start - 2,
#     ando1_start + 2,
#     resolution=0.05,
#     GPIB_num=[GPIB_val, 19],
#     sweeptype="SGL",
# )
# |%%--%%| <sBLPSnboxw|xtu3fRqC9j>
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
        edfa.activeON()


start_edfa(edfa1, EDFA_power[0])
#%%

scope = TektronixOscilloscope()
# |%%--%%| <xtu3fRqC9j|ZJKYvV31wM>
pulse_freqs = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10]) * 10**5
# pulse_freqs = np.array([0.5, 1]) * 10**5
# pulse_freq = pulse_freqs[3]
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75]
time_per_pulse = 1 / pulse_freqs
num_pulses = 10
time_axis0, voltage0 = scope.saveWaveform(1)
dict_keys = ['duty_cycle', 'time_ax_ref', 'voltage_ref', 'time_ax', 'voltage']
scope_data_dict = {f"{int(pulse_freq * 10**-3)}KHz": {sub_key: [] for sub_key in dict_keys} for pulse_freq in pulse_freqs}
scope_data_keys = list(scope_data_dict.keys())
scope.setTrigger(1, type="edge")
for j, pulse_freq in enumerate(pulse_freqs):
    super_key = scope_data_keys[j]
    scope.setTimeScale(time_per_pulse[j])
    for i, duty in enumerate(duty_cycle):
        pico.awg.set_square_wave(pulse_freq, duty)
        time.sleep(0.5)
        try:
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
        except PicoSDKCtypesError:
            time_axis_ref = np.nan * np.ones(1000)
        if duty == 1:
            scope.trigger(1, type="auto")
        print("duty cycle: ", duty)
        scope.wait()
        time_axis, voltage = scope.saveWaveform(1)
        scope.wait()
        scope_data_dict[super_key]["duty_cycle"].append(duty)
        scope_data_dict[super_key]["time_ax_ref"].append(time_axis_ref)
        scope_data_dict[super_key]["voltage_ref"].append(y_ref)
        scope_data_dict[super_key]["time_ax"].append(time_axis)
        scope_data_dict[super_key]["voltage"].append(voltage)
# scope.trigger(1, type="edge")
#%%
import pickle

with open("C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/scope_traces/freq_and_cycle_vary.pkl", "wb") as f:
    pickle.dump(scope_data_dict, f)
#|%%--%%| <bY9mjYPvjG|RPoBAOYwJ0>
time_axis, voltage = scope.saveWaveform(4, stop_val=200000)
trace = np.array([time_axis, voltage])
plt.plot(time_axis, voltage)
plt.show()
folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/scope_traces/sig_idler_traces"
pump_sep = 15
if not os.path.exists(folder):
    os.makedirs(folder)
with open(f"{folder}/pump_sep_{pump_sep}_sig.pkl", "wb") as f:
    pickle.dump(trace, f)
#|%%--%%| <dsC6e861y3|OhDxuUlLXi>
freq_idx = 4
sup_key = list(scope_data_dict.keys())[freq_idx]
duty_idx = 3
fig, ax = plt.subplots(1, 1)
ax.plot(
    scope_data_dict[sup_key]["time_ax"][duty_idx],
    scope_data_dict[sup_key]["voltage"][duty_idx],
    label="duty cycle = " + str(duty_cycle[duty_idx]),
)

# for i in range(len(duty_cycle)):
#     ax.plot(
#         scope_data_dict[sup_key]["time_ax"][i],
#         scope_data_dict[sup_key]["voltage"][i],
#         label="duty cycle = " + str(duty_cycle[i]),
#     )

