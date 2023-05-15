import numpy as np
import time
import ctypes
import matplotlib.pyplot as plt
import sys

sys.path.append("U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl")
from picoscope2000 import PicoScope2000a

pico_scope = PicoScope2000a()
# |%%--%%| <tyF7do8SbO|tJEz7J6Y8a>
start_frequency = 1 * 10**5
waveform_size = 2**12
waveform = (ctypes.c_int16 * waveform_size)()
duty_cycle = 0.5
threshold = int(waveform_size * duty_cycle)
for i in range(waveform_size):
    if i < threshold:
        waveform[i] = 32767
    else:
        waveform[i] = -32767

pico_scope.awg.set_params(
    start_frequency, waveform_data=waveform, pk_to_pk=2 * 10**6, offset=1 * 10**6
)
# pico_scope.awg.set_params(start_frequency, wave_type=1, pk_to_pk=2 * 10**6, offset=1 * 10**6)
# pico_scope.awg.set_params(start_frequency, wave_type=1, pk_to_pk=2 * 10**6)
# pico_scope.awg.set_params(start_frequency, wave_type=1)
pico_scope.oscilloscope.set_params(
    channel=1,
    enabled=0,
)
pico_scope.oscilloscope.set_params(
    channel=0,
    enabled=1,
    channel_range=7,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico_scope.oscilloscope.set_sampling_rate(start_frequency, peaks_wanted=2)
time_axis1, y1, overflow, adc1, pulse_length, pulse_dist = pico_scope.oscilloscope.run_scope(threshold_high=0.5, threshold_low=0.5)
print("Pulse length: ", pulse_length)
print("Pulse distance: ", pulse_dist)
pico_scope.oscilloscope.set_params(
    channel=0,
    enabled=0,
)
pico_scope.oscilloscope.set_params(
    channel=1,
    enabled=1,
    channel_range=4,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico_scope.oscilloscope.set_sampling_rate(start_frequency, peaks_wanted=2)
time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = pico_scope.oscilloscope.run_scope()
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.plot(time_axis, y)
ax.set_title(f'Pulse length: {pulse_length} us, Pulse distance: {pulse_dist} us')
ax1.plot(time_axis1, y1, color='orange')
ax.set_xlabel("Time (us)")
ax.set_ylabel("Voltage (mV)")
# plt.savefig('./data/michael/high_CE_betweenpumps_1e5Hz_10duty_pulses.png')
plt.show()
