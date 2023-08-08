import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.ion()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

with open('../data/scope_traces/freq_and_cycle_vary.pkl', 'rb') as f:
    data = pickle.load(f)
#|%%--%%| <kmVhiYHnNW|t2yhxHMzWr>
freqs = list(data.keys())
freq_idx = 0
freq = freqs[freq_idx]
sub_data = data[freq]
duty_cycles = np.array(sub_data['duty_cycle'])
time_ref = np.array(sub_data['time_ax_ref'])
voltage_ref = np.array(sub_data['voltage_ref'])
time = np.array(sub_data['time_ax'])
voltage = np.array(sub_data['voltage'])

fig, ax = plt.subplots()
for i, duty_cycle in enumerate(duty_cycles):
    # ax1 = ax.twinx()
    # ax1.plot(time_ref[i], voltage_ref[i], label='Reference')
    ax.plot(time[i] * 10**6, voltage[i] * 10**3, label=f'{duty_cycle*100:.0f}%')
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title(f'Frequency={freq}')
    ax.legend()
#|%%--%%| <t2yhxHMzWr|9AR6e6CFKs>
freq_idx = 0
freq = freqs[freq_idx]
sub_data = data[freq]
duty_cycles = np.array(sub_data['duty_cycle'])
time_ref = np.array(sub_data['time_ax_ref'])
voltage_ref = np.array(sub_data['voltage_ref'])
time = np.array(sub_data['time_ax'])
voltage = np.array(sub_data['voltage'])

for i, duty_cycle in enumerate(duty_cycles):
    fig, ax = plt.subplots()
    # ax1 = ax.twinx()
    # ax1.plot(time_ref[i], voltage_ref[i], label='Reference')
    ax.plot(time[i] * 10**6, voltage[i] * 10**3)
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title(f'Frequency={freq}, Duty Cycle: {duty_cycle}')
    ax.legend()
