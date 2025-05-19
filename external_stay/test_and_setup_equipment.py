import enum
import os
import re
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from clients.dc_power_supply_clients import E3631A
from clients.diode_controller_clients import EtekMLDC1032
from clients.hp81200_client import HP81200Client, HP81200ModuleClient
from clients.laser_clients import AgilentLaserClient
from clients.osa_clients import OSAClient
from clients.oscilloscope_clients import HP83480
from clients.power_meter_clients import Agilent8163x
from clients.rf_clock_clients import HP8341
from clients.scope_Agilent3000Xseries_client import Agilent3000X
from processing.helper_funcs import (
    dBm_to_mW,
    mW_to_dBm,
    Fiber,
    get_fiber,
    load_fibers_from_json,
)
from external_stay.run_exp_funcs.bias import (
    get_dc_voltages,
    set_dc_voltages,
    increment_bias,
    sweep_bias_get_scope_traces,
    plot_mzm_biases,
    sweep_all_dc_supplies,
    BiasSweepIdxDepVars,
    BiasSweepGeneralVars,
    sweep_scope_multiple_reps,
    sweep_bias,
    start_scrambler,
    stop_scrambler,
    toggle_scrambling,
)
from external_stay.run_exp_funcs.helper_funcs import (
    center_scope_around_peak,
    center_scope_around_position,
    center_scope_around_max_val,
    extract_sequence_names,
    find_square_region,
    get_scope_data,
    set_multiple_seqs,
    set_one_seq,
    turn_off_edfas,
    turn_on_edfas,
    get_osa_spec,
    increment_rf_clock,
    get_n_scope_traces,
    get_ones_pos,
    increment_segment,
    increment_multiple_segments,
    set_vertical_bars_around_peak_scope,
    optimize_brill_freq_hill_climb,
    optimize_brill_freq_golden,
    calc_segment_len_m,
    calc_pulse_len,
    DiodeControllerRouter,
    print_time_status,
    get_dark_currents,
)
from external_stay.run_exp_funcs.brill_amp import (
    TracesSpectraMultipleColls,
)
from external_stay.run_exp_funcs.moving_collisions import (
    MovingCollisionInputParams,
    InstrumentSetup,
    CollisionMeasurementData,
    full_filepath_collision_meas,
    run_collision_sweep,
)


plt.style.use("custom")
plt.ioff()
# |%%--%%| <q9sfVDPyr8|F8E7tZ5vIK>
base_url = "http://100.80.225.40:5000"
base_url_hp81200 = "http://100.80.225.106:5000"
osa = OSAClient(
    base_url,
    "osa_1",
    (1554.75, 1555.25),
    resolution=0.05,
    GPIB_address=25,
    GPIB_bus=0,
    sweeptype="RPT",
    zero_nm_sweeptime=50,
)
wl1_init = 1530
wl2_init = 1560
tls1 = AgilentLaserClient(
    base_url,
    "agilent_laser_1",
    target_wavelength=wl1_init,
    power=2,
    source=0,
    GPIB_address=21,
)
tls2 = AgilentLaserClient(
    base_url,
    "agilent_laser_2",
    target_wavelength=wl2_init,
    power=2,
    source=2,
    GPIB_address=20,
)
tls1.wavelength = wl1_init
tls2.wavelength = wl2_init
tls1.enable()
tls2.enable()
pm_1 = Agilent8163x(base_url, "agilent_pm_1", 1550, channel=2, GPIB_address=21)
pm_2 = Agilent8163x(base_url, "agilent_pm_2", 1550, channel=3, GPIB_address=21)
pm_3 = Agilent8163x(base_url, "agilent_pm_3", 1550, channel=4, GPIB_address=21)
rf_clock = HP8341(base_url, "hp8341", frequency=9.2e9, power=-6, GPIB_address=19)
scope = HP83480(base_url, "hp83480", "ASCII", GPIB_address=8)
small_scope = Agilent3000X(base_url, "DSOX", "WORD", 1)
dc_1 = E3631A(
    base_url,
    "e3631a_1",
    connected_channels=[2, 3],
    current_limit=0.01,
    limit_all_channels_on_init=True,
    GPIB_address=1,
)
# Channel controlling pol scrambler
dc_1.cur_channel = 1
dc_1.current_limit = 1.5
# Controlling 1 and 2
dc_2 = E3631A(
    base_url,
    "e3631a_2",
    connected_channels=[2],
    current_limit=0.01,
    limit_all_channels_on_init=True,
    GPIB_address=2,
)
dc_3 = E3631A(
    base_url,
    "e3631a_3",
    connected_channels=[2],
    current_limit=0.01,
    limit_all_channels_on_init=True,
    GPIB_address=4,
)
dc_4 = E3631A(
    base_url,
    "e3631a_3",
    connected_channels=[1, 2, 3],
    current_limit=0.01,
    limit_all_channels_on_init=True,
    GPIB_address=3,
)
edfas_1 = {
    "EDFA2": [14, 21, 22],
    "PRE1": [11, 12],
    "EDFAC3": [19, 20],
    "PRE0": [3, 4],
    "PRE2_1": [27, 28],
}
edfas_2 = {
    "EDFA1": [3, 5, 9, 13],
    "PRE3": [1],
    "PRE2_2": [7, 11],
}
mA_lims_1 = {
    3: 250,
    4: 250,
    11: 250,
    12: 500,
    14: 500,
    19: 280,
    20: 280,
    21: 220,
    22: 220,
    27: 350,
    28: 350,
}

mA_lims_2 = {
    1: 150,
    3: 750,
    5: 250,
    7: 250,
    9: 250,
    11: 250,
    13: 750,
}
diode_controller_1 = EtekMLDC1032(
    base_url, "etek_mldc_1", edfas=edfas_1, mA_lims=mA_lims_1, GPIB_address=6
)
diode_controller_2 = EtekMLDC1032(
    base_url, "etek_mldc_2", edfas=edfas_2, mA_lims=mA_lims_2, GPIB_address=7
)


diode_controller = DiodeControllerRouter([diode_controller_1, diode_controller_2])


# |%%--%%| <Xa5ZD2vGWw|0phiLjkr0O>
pulse_gen_freq = 1e9
seg_len_m = calc_segment_len_m(pulse_gen_freq)
hp81200_base = HP81200Client(
    base_url_hp81200, "hp81200", num_ports=5, frequency=pulse_gen_freq
)
hp81200_base.blg_factor = 32
voltage_levels_1 = [-0.3, 0.3]
voltage_levels_2 = [-0.2, 0.3]
voltage_levels_3 = [-0.1, 0.3]
voltage_levels_4 = [-0.3, 0.3]
voltage_levels_5 = [-0.1, 0.3]
generator_1 = HP81200ModuleClient(
    base_url_hp81200,
    "hp81200_module_1",
    "HP81200",
    module_id=1,
    voltage_levels=voltage_levels_1,
)
generator_2 = HP81200ModuleClient(
    base_url_hp81200,
    "hp81200_module_2",
    "HP81200",
    module_id=2,
    voltage_levels=voltage_levels_2,
)
generator_3 = HP81200ModuleClient(
    base_url_hp81200,
    "hp81200_module_3",
    "HP81200",
    module_id=3,
    voltage_levels=voltage_levels_3,
)

generator_4 = HP81200ModuleClient(
    base_url_hp81200,
    "hp81200_module_4",
    "HP81200",
    module_id=4,
    voltage_levels=voltage_levels_4,
)
generator_5 = HP81200ModuleClient(
    base_url_hp81200,
    "hp81200_module_5",
    "HP81200",
    module_id=5,
    voltage_levels=voltage_levels_5,
)
seq_len = 2**12

hp81200_base.set_sequences(
    ["thjalfe1", "thjalfe2", "thjalfe3", "thjalfe4", "thjalfe5"],
    sequence_length=seq_len,
    num_generators=5,
)
# |%%--%%| <2EUNnWLj06|KTSnUxBKaz>
# HP81200 section
seq_len = 2**12
# seq_len = 10000
pulselen_1 = 15
pulselen_2 = 10
pulselen_3 = 10
pulselen_4 = pulselen_1 + 1
pulselen_5 = pulselen_2 + 1
# pulselen_4 = 100
idx_from_1_to_4 = 31
idx_from_2_to_5 = 10
# start_idx_1 = 2860  # HNDS1615AAA
start_idx_1 = 2165  # HNDS1599CA
# start_idx_1 = 1804  # HNDS1599CA
# start_idx_1 = 2610
start_idx_2 = 1635
start_idx_3 = start_idx_2 + 100
start_idx_4 = start_idx_1 + idx_from_1_to_4
start_idx_5 = start_idx_2 + idx_from_2_to_5
seq1 = np.arange(start_idx_1, start_idx_1 + pulselen_1, 1)
seq2 = np.arange(start_idx_2, start_idx_2 + pulselen_2, 1)
seq3 = np.arange(start_idx_3, start_idx_3 + pulselen_3, 1)
seq4 = np.arange(start_idx_4, start_idx_4 + pulselen_4, 1)
seq5 = np.arange(start_idx_5, start_idx_5 + pulselen_5, 1)
seqs = [seq1, seq2, seq3, seq4, seq5]


set_multiple_seqs(hp81200_base, seq_len, seqs)
# |%%--%%| <9sz0sKzbGx|7RviKjXDNy>
new_pulselen = 17
id = 4
cur_seq = get_ones_pos(hp81200_base, id)
newseq = np.arange(cur_seq[0], cur_seq[0] + new_pulselen)
set_one_seq(hp81200_base, id, newseq, seq_len, num_segments=5)


# |%%--%%| <7RviKjXDNy|XFHuXNsm2p>
def time_in_fiber(fiber_len: float, c_fiber=2e8):
    return fiber_len / c_fiber


def calc_pulse_duration(pulse_len: float, c_fiber=2e8) -> float:
    return pulse_len / c_fiber


def calc_seq_len_for_one_pulse_in_fiber(
    fiber_len: float, freq: float = 1e9, c_fiber=2e8
) -> int:
    time_in_fib = time_in_fiber(fiber_len, c_fiber)
    return int(np.ceil(freq * time_in_fib))


# |%%--%%| <XFHuXNsm2p|OGEPYTWvFa>
x, y = get_scope_data(scope, 1)
print(np.max(y))
plt.plot(x, y)
plt.show()
# |%%--%%| <OGEPYTWvFa|Vu0P2hn97r>
# SECTION FOR SWEEPING BIAS OF MZM
voltages = np.arange(0, 20 + 0.1, 0.1)
# voltages = np.arange(0, 20 + 0.1, 10)
mzm_dict = {
    "wl": 1550,
    "power": 0,
    "mzm_1": np.zeros((2, len(voltages))),
    "mzm_2": np.zeros((2, len(voltages))),
    "mzm_3": np.zeros((2, len(voltages))),
}


dc_channels = [2, 2, 3]
dc_lst = [dc_2, dc_1, dc_1]
pms = [pm_1, pm_2, pm_3]
mzm_dict = sweep_all_dc_supplies(mzm_dict, dc_lst, pms, dc_channels, voltages)
data_dir = "data/mzm_bias_sweep/no_rf_in"
save_data = True
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if save_data:
    with open(f"{data_dir}/mzm_bias_sweep.pkl", "wb") as f:
        pickle.dump(mzm_dict, f)

plot_mzm_biases(mzm_dict, data_dir, save_data, show_plots=True)

# |%%--%%| <tcsIPZOvEJ|rI3OE7N45E>
# SCOPE MEASUREMENTS
data_dir = "./data/SSB-mod/optimizing_pulse"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
filename = "scope_traces_pre-amp-before-SSB-after-MZM_wfilter-before-largeEDFA"
filename = f"{data_dir}/{filename}"
num_reps = 10
traces = sweep_scope_multiple_reps(scope, 10, 1)
with open(f"{filename}.pkl", "wb") as handle:
    pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)
# |%%--%%| <rI3OE7N45E|g0wbyIZbpS>
# Check power of different sequence lengths of pump
reps_for_each_seq_len = 100
pulse_lens = np.arange(0, 101, 5)
seq_len = 5070
idx = 2
seq = np.arange(start_idx_1, start_idx_1 + pulse_lens[idx], 1)  # 1
set_one_seq(hp81200_base, 1, seq, seq_len)
# |%%--%%| <OYWo9yNvqn|nnzs1nFE6o>
powers = []
for j in range(reps_for_each_seq_len):
    time.sleep(0.2)
    powers.append(pm_3.power)
print(np.mean(powers))
print(np.std(powers))
# |%%--%%| <nnzs1nFE6o|4OkGMhTzFF>
powers = np.zeros((len(pulse_lens), reps_for_each_seq_len))
time_traces = []
for idx, pulse_len in enumerate(pulse_lens):
    if idx == 0:
        hp81200_base.output = False
        for j in range(reps_for_each_seq_len):
            time.sleep(0.2)
            powers[idx][j] = pm_3.power
        hp81200_base.output = True
    else:
        seq = np.arange(start_idx_1, start_idx_1 + pulse_len, 1)  # 1
        set_one_seq(hp81200_base, 1, seq, seq_len)
        for j in range(reps_for_each_seq_len):
            time.sleep(0.2)
            powers[idx][j] = pm_3.power
        scope.timebase.scale = int(np.round(pulse_len * 0.5)) * 10**-9
        time.sleep(0.5)
        trace = get_scope_data(scope, 3)
        time_traces.append(trace)
    print(f"Done with {idx+1} out of {len(pulse_lens)}")
# |%%--%%| <4OkGMhTzFF|k8ZRplIFXK>
mean_powers = np.mean(powers, axis=1)
std_powers = np.std(powers, axis=1)
fig, ax = plt.subplots()
ax.errorbar(
    pulse_lens, mean_powers * 1000, yerr=std_powers * 1000, markersize=10, capsize=2
)
ax.set_xlabel("Pulse duration [ns]")
ax.set_ylabel("Average power [mW]")
plt.show()
# |%%--%%| <f7Gv8jVcGb|8u9rh91eQ9>
power_subset = mean_powers[2:-4]
pulse_len_sub = pulse_lens[2:-4] * 10**-9

# |%%--%%| <k8ZRplIFXK|f7Gv8jVcGb>
# Sweep bias while measuring the response on the scope

data_dir = "./data/mzm_bias_sweep/rf_in"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
filename = "scope-traces_bias-varied-close-to-peak_all-3-mzms_10ns-pulses"
# with open(f"{data_dir}/{filename}.pkl", "rb") as f:
#     trace_data_dict = pickle.load(f)
save_data = True
dc_channels = [2, 2, 3]
dc_lst = [dc_2, dc_1, dc_1]
scope_channels = [1, 1, 4]
dc_start_voltages = [3.7, 14, 10.26]
num_reps = 3
delta_voltage = 8
voltage_stepsize = 0.2
trace_name_lst = ["trace_1", "trace_2", "trace_3"]
sweep_vars = [
    BiasSweepIdxDepVars(
        dc_channel=dc_channels[0],
        dc=dc_lst[0],
        dc_start_voltage=dc_start_voltages[0],
        scope_channel=scope_channels[0],
    ),
    BiasSweepIdxDepVars(
        dc_channel=dc_channels[1],
        dc=dc_lst[1],
        dc_start_voltage=dc_start_voltages[1],
        scope_channel=scope_channels[1],
    ),
    BiasSweepIdxDepVars(
        dc_channel=dc_channels[2],
        dc=dc_lst[2],
        dc_start_voltage=dc_start_voltages[2],
        scope_channel=scope_channels[2],
    ),
]
sweep_vars = [sweep_vars[1]]
trace_name_lst = [trace_name_lst[1]]
gen_vars = BiasSweepGeneralVars(
    num_reps=num_reps,
    delta_voltage=delta_voltage,
    voltage_stepsize=voltage_stepsize,
    trace_name_lst=trace_name_lst,
)
trace_dict = sweep_bias_get_scope_traces(
    scope,
    sweep_vars,
    gen_vars,
    data_dir,
    filename,
    save_data,
    sleeptime=0.25,
    pulse_window=5070e-9,
)
# |%%--%%| <8u9rh91eQ9|D6N5wX5Q5C>
# Sweep bias to get DC response to find p2p
save_dir = "./data/mzm_bias_sweep/no_rf_in"
filename = "coarse_sweep_find_p2p_codion-mach-10_idler-arm.csv"
voltage_array = np.arange(0, 15, 0.1)
sleeptime = 0.2
data = sweep_bias(voltage_array, pm_3, dc_3, 1, sleeptime)
# np.savetxt(f"{save_dir}/{filename}", data)
# data = np.loadtxt(f"{save_dir}/{filename}")
# |%%--%%| <D6N5wX5Q5C|eFIRBuSo2t>
fig, ax = plt.subplots()
ax.plot(data[0], data[1])
fig, ax = plt.subplots()
ax.plot(data[0], dBm_to_mW(data[1]))
plt.show()
# |%%--%%| <eFIRBuSo2t|jyqdTMWyUQ>
# Section for measuring walk-off between pulses in fiber
fiber_name = "HNDS1615AAA-6-3-2"

fiber_len = "250m"
data_dir = f"./data/walk-off/{fiber_name}"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
wavelengths = np.arange(1525, 1566, 1)
filename = f"{wavelengths[0]:.1f}nm-{wavelengths[-1]:.1f}nm_scope-traces_{fiber_len}"
trace = get_scope_data(scope, 1)
trace_dict = {
    "traces": np.zeros((len(wavelengths), np.shape(trace)[0], np.shape(trace)[1])),
    "wavelengths": wavelengths,
    "IL": 1.7,
}
for idx, wl in enumerate(wavelengths):
    tls1.wavelength = float(wl)
    time.sleep(0.5)
    trace = get_scope_data(scope, 1)
    trace_dict["traces"][idx] = trace
with open(f"{data_dir}/{filename}.pkl", "wb") as f:
    pickle.dump(trace_dict, f)
# np.savetxt(f"{data_dir}/{filename}.csv", trace)
# |%%--%%| <pIYfjLgjw6|Y2gcrp8XMi>
idxs_to_plot = np.arange(0, len(wavelengths), 10)
fig, ax = plt.subplots(nrows=1, ncols=1)
for idx in idxs_to_plot:
    ax.plot(
        trace_dict["traces"][idx, 0, :],
        trace_dict["traces"][idx, 1, :] / np.max(trace_dict["traces"][idx, 1, :]),
        label=f"{wavelengths[idx]}",
    )
ax.legend()
ax.set_xlabel(r"")
ax.set_ylabel(r"")
plt.show()
# |%%--%%| <3d9xDw98JV|R4dJJNCFT3>
# SSB from OSA
data_dir = "./data/SSB-mod"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
freq = rf_clock.frequency
volt = rf_clock.power
assert isinstance(freq, float)
assert isinstance(volt, float) or isinstance(volt, int)
filename = f"suppresion-spectrum_freq={freq*10**-9:.3f}GHz_rf-power={volt:.1f}dBm_loss=13.2dB_1"

filename = f"{data_dir}/{filename}"


def increment_filename(filename):
    if filename.endswith(".pkl"):
        filename = filename[:-4]

    parts = filename.split("_")
    match = re.match(r"(\d+)$", parts[-1])
    if match:
        counter = int(match.group(1))
        base = "_".join(parts[:-1])
    else:
        counter = 1
        base = "_".join(parts)

    # Increment until a unique filename is found
    new_filename = f"{base}_{counter}.pkl"
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}.pkl"

    return new_filename


filename = increment_filename(filename)

# bias_voltages = [5.49, -2.9, 4.642]
bias_voltages = [3.47, -3.31, 5.142]
bias_voltages = [3.33, -3.26, 5.053]
osa.sweeptype = "SGL"
osa.sweep()
spectrum = np.vstack((osa.wavelengths, osa.powers))
data = {"biases": bias_voltages, "spectrum": spectrum}
with open(f"{filename}.pkl", "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

osa.sweeptype = "RPT"
osa.sweep()


# |%%--%%| <R4dJJNCFT3|dFLUo7WHAf>
# Brillouin gain


save_data = False
data_dir = "./data/brillioun_amp"
stepsize = 10e6
start_freq = 9e9
stop_freq = 10e9
rf_freqs = np.arange(start_freq, stop_freq + 1, stepsize)
filename = f"spectra_const-bias-optimized-for-9.5GHz_rf-freq={start_freq*10**-9:.1f}GHz-{stop_freq*10**-9:.1f}GHz_stepsize={stepsize*10**-6:.1f}MHz"
filename = f"{data_dir}/{filename}"
data_dict = {
    "spectra": [],
    "rf_freqs": rf_freqs,
    "bias_voltages": bias_voltages,
    "rf_pow": rf_clock.power,
}
osa.sweeptype = "SGL"


for freq in rf_freqs:
    rf_clock.frequency = freq
    time.sleep(1)
    osa.sweep()
    spectrum = get_osa_spec(osa)
    data_dict["spectra"].append(spectrum)

if save_data:
    with open(f"{filename}.pkl", "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# |%%--%%| <yKcbyoSbHN|ATynY6i83G>
save_data = True
data_dir = "./data/brillioun_amp/rf_sweep-manual-bias"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
filename = f"rf-freq={rf_freqs[0]*10**-9:.2f}GHz-{rf_freqs[-1]*10**-9:.2f}GHz_stepsize={np.diff(rf_freqs)[0]*10**-6:.2f}MHz_pulse-len=10ns_single-pulse-in-fiber"
filename = f"{data_dir}/{filename}"
num_reps = 5
osa.sweeptype = "SGL"
osa.sweep()
dummy = get_osa_spec(osa)
data_dict = {
    "rf_freqs": rf_freqs,
    "biases": voltages,
    "pump_ref": None,
    "ref_spectra": np.zeros((len(rf_freqs), 2, np.shape(dummy)[1])),
    "amplified_spectra": np.zeros((len(rf_freqs), num_reps, 2, np.shape(dummy)[1])),
}
diode_controller_2.set_edfa_to_max("EDFA1")
diode_controller_1.disable_edfa("EDFA2")
osa.sweeptype = "SGL"
osa.sweep()
spec = get_osa_spec(osa)
data_dict["pump_ref"] = spec
for i in range(2):
    if i == 0:
        cur_spec_key = "ref_spectra"
        diode_controller_2.disable_edfa("EDFA1")
        diode_controller_1.set_edfa_to_max("EDFA2")
        ref = True
    else:
        cur_spec_key = "amplified_spectra"
        diode_controller_2.set_edfa_to_max("EDFA1")
        ref = False
    for j, rf_freq in enumerate(rf_freqs):
        set_dc_voltages(dc_3, voltages[j])
        if not ref:
            for k in range(num_reps):
                rf_clock.frequency = rf_freq
                osa.sweep()
                spec = get_osa_spec(osa)
                data_dict[cur_spec_key][j, k] = spec
        else:
            rf_clock.frequency = rf_freq
            osa.sweep()
            spec = get_osa_spec(osa)
            data_dict[cur_spec_key][j] = spec
if save_data:
    with open(f"{filename}.pkl", "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# |%%--%%| <WfqIfjCrg0|383PSy4qmj>
#
# Brill gain time traces
save_data = True
data_dir = "./data/brillioun_amp/time_traces"
filename = "traces_with-and-without-amplification"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_dict = {"rf_freq": rf_clock.frequency, "amplified_traces": [], "ref_traces": []}
num_reps = 21
for i in range(num_reps):
    trace = get_scope_data(scope, 1)
    data_dict["amplified_traces"].append(trace)
data_dict["amplified_traces"] = np.array(data_dict["amplified_traces"])
diode_controller_2.disable_edfa("EDFA1")
for i in range(num_reps):
    trace = get_scope_data(scope, 1)
    data_dict["ref_traces"].append(trace)
data_dict["ref_traces"] = np.array(data_dict["ref_traces"])
if save_data:
    with open(f"{data_dir}/{filename}.pkl", "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# |%%--%%| <383PSy4qmj|mgg9Ppxmji>
# Brill gain time traces + spectra
fiber_dict = load_fibers_from_json()
fiber = get_fiber("fiber_2", fiber_dict)


def base_dir_path_brill_amp(
    fiber: Fiber,
    base_path: str = "/home/thjalfe/Documents/PhD/Projects/Experiments/external_stay/data/brillioun_amp",
):
    fiber_specific = f"Fiber={fiber.full_fiber_name}_{fiber.length:.0f}meter"
    path = f"{base_path}/{fiber_specific}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# |%%--%%| <mgg9Ppxmji|WUWefAGjca>
# Improved version of the above code to include possibility of moving pulses as well
# extra_filename = "two-preamps-collision-start-of-fiber-very-high-gain_maximizingFWM"
save_data = True
base_dir = base_dir_path_brill_amp(fiber)
full_dir_path = f"{base_dir}/weird_pulse_things"
if not os.path.exists(full_dir_path):
    os.makedirs(full_dir_path)
extra_filename = "full-seq-len=2km_moving_pump_const-pol"
rf_freq = rf_clock.frequency
assert isinstance(rf_freq, float)
full_filename = (
    f"{full_dir_path}/rf={rf_freq*10**-9:.2f}GHz_traces-and-spectra_{extra_filename}"
)


num_reps_for_spectra = 11
dummy_trace = get_scope_data(scope, 1)
trace_shape = np.shape(dummy_trace)
osa.update_spectrum()
dummy_osa_spec = get_osa_spec(osa)
segment_stepsize = 10
num_segment_steps = 10
data_class = TracesSpectraMultipleColls(
    num_reps_for_spectra,
    np.shape(dummy_trace),
    np.shape(dummy_osa_spec),
    num_segment_steps,
)

sig_scope_pos = 2.36596e-05
pump_scope_pos = 2.4928e-05

data_class.traces_and_spectra_multiple_collisions_meas(
    scope,
    osa,
    diode_controller_1,
    diode_controller_2,
    hp81200_base,
    sig_scope_pos,
    pump_scope_pos,
    segment_stepsize,
)

data_class.save(f"{full_filename}_2.pkl")


# |%%--%%| <WUWefAGjca|wCAiJx117s>
def fix_discontinuity(y: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Fixes step discontinuity in a waveform by detecting sudden jumps and subtracting the offset after the jump.
    Parameters:
        x: time axis
        y: voltage signal
        threshold: minimum jump to count as discontinuity [V]
    Returns:
        Corrected y array
    """
    y = np.asarray(y)
    original_mean = np.mean(y)
    original_std = np.std(y)

    if original_std == 0:
        return y  # constant signal, nothing to fix

    # Normalize
    y_norm = (y - original_mean) / original_std
    dy_norm = np.diff(y_norm)
    jumps = np.where(np.abs(dy_norm) > threshold)[0]

    if len(jumps) == 0:
        return y  # no discontinuity detected

    corrected_y_norm = y_norm.copy()
    for jump_idx in jumps:
        offset = corrected_y_norm[jump_idx + 1] - corrected_y_norm[jump_idx]
        corrected_y_norm[jump_idx + 1 :] -= offset

    # Rescale back
    corrected_y = corrected_y_norm * original_std + original_mean
    return corrected_y


def pulse_area(
    x: np.ndarray, y: np.ndarray, threshold: float = 0.1, tail_fraction: float = 0.01
) -> float:
    y = np.asarray(y)
    x = np.asarray(x)

    n = len(y)
    tail_len = int(n * tail_fraction)

    # Estimate and remove DC baseline
    baseline = np.mean(np.concatenate([y[:tail_len], y[-tail_len:]]))
    y_corr = y - baseline

    # Find pulse region
    max_val = np.max(y_corr)
    assert isinstance(max_val, float)
    threshold = threshold * max_val
    mask = y_corr > threshold
    if not np.any(mask):
        return 0.0
    dx = x[1] - x[0]
    return np.sum(y_corr[mask]) * dx


# |%%--%%| <wCAiJx117s|gSYpbpv8iW>


def optimize_brill_freq_hill_climb_scope(
    scope: Agilent3000X,
    scope_channel: int,
    rf_clock: HP8341,
    initial_step: float = 10e6,
    min_step: float = 1e6,
    print_powers=False,
) -> float:
    current_freq = rf_clock.frequency
    assert isinstance(current_freq, float)
    step = initial_step
    scope.waveform.channel = scope_channel

    # Initial scan: measure at f - step, f, f + step
    trial_freqs = [
        round((current_freq - np.abs(step)) / 1e5) * 1e5,
        round(current_freq / 1e5) * 1e5,
        round((current_freq + np.abs(step)) / 1e5) * 1e5,
    ]
    trial_volts = []

    for f in trial_freqs:
        rf_clock.frequency = f
        x, y = scope.waveform.read_waveform()
        y = fix_discontinuity(y)
        volt = pulse_area(x, y)
        trial_volts.append(volt)

    best_index = int(np.argmax(trial_volts))
    current_freq = trial_freqs[best_index]
    current_power = trial_volts[best_index]

    # Determine direction: are we going up or down in freq?
    if best_index == 0:
        step = -abs(step)
    elif best_index == 2:
        step = abs(step)
    else:
        # we were already at the best spot — arbitrarily go positive
        step = abs(step)
    freqs = []
    pow = []

    while abs(step) > min_step:
        next_freq = round((current_freq + step) / 1e5) * 1e5
        rf_clock.frequency = next_freq
        x, y = scope.waveform.read_waveform()
        y = fix_discontinuity(y)
        volt = pulse_area(x, y)
        assert isinstance(volt, float)
        if print_powers:
            print(volt)

        if volt > current_power:
            current_freq = next_freq
            current_power = volt
            step *= 1.1
        else:
            step *= -0.5

        freqs.append(next_freq)
        pow.append(current_power)

    max_freq = freqs[np.argmax(pow)]
    rf_clock.frequency = max_freq
    return max_freq


def optimize_brill_freq_golden_scope(
    scope: Agilent3000X,
    scope_channel: int,
    rf_clock: HP8341,
    low: float,
    high: float,
    tol: float = 1e6,
    print_trace: bool = False,
) -> float:
    invphi = (np.sqrt(5) - 1) / 2  # 0.618…
    invphi2 = (3 - np.sqrt(5)) / 2  # 0.382…
    scope.waveform.channel = scope_channel

    a, b = low, high
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)

    def measure(f):
        rf_clock.frequency = f
        x, y = scope.waveform.read_waveform()
        y = fix_discontinuity(y)
        volt = pulse_area(x, y)
        return volt

    Pc, Pd = measure(c), measure(d)
    history = [(c, Pc), (d, Pd)]

    while abs(b - a) > tol:
        if Pc > Pd:
            b, Pd = d, Pc
            d = c
            c = a + invphi2 * (b - a)
            Pc = measure(c)
            history.append((c, Pc))
        else:
            a, Pc = c, Pd
            c = d
            d = a + invphi * (b - a)
            Pd = measure(d)
            history.append((d, Pd))
        if print_trace:
            print(f"a={a:.3e}, b={b:.3e}")
    # choose the best sampled point
    best_f, best_P = max(history, key=lambda x: x[1])
    rf_clock.frequency = best_f
    return best_f


rf_clock.frequency = 9.19e9
t0 = time.time()
start = rf_clock.frequency
assert isinstance(start, float)
brackets = 20e6
opt_f = optimize_brill_freq_hill_climb_scope(
    small_scope,
    1,
    rf_clock,
    initial_step=20e6,
    min_step=1e6,
    print_powers=True,
)
# opt_f = optimize_brill_freq_golden_scope(
#     small_scope,
#     1,
#     rf_clock=rf_clock,
#     low=start - brackets,
#     high=start + brackets,
#     tol=1e6,
#     print_trace=False,
# )
# opt_f = optimize_brill_freq_hill_climb(
#     pm_2, rf_clock, initial_step=20e6, min_step=1e6, delay=0.1, print_powers=True
# )
# opt_f = optimize_brill_freq_golden(
#     pm=pm_2,
#     rf_clock=rf_clock,
#     low=start - brackets,
#     high=start + brackets,
#     tol=1e6,
#     delay=0.1,
#     print_trace=False,
# )

print(opt_f)
print(time.time() - t0)

# |%%--%%| <ZA4eB1YnDT|0Q11mNLPpr>
from dataclasses import dataclass, field


@dataclass
class MovingCollisionInputParamsScope:
    pulse_gen_freq: float
    brill_pump_seg_len: int
    pump_seg_len: int
    stepsize_seg: int
    start_idx: int
    end_idx: int
    brill_freq_init_step: float
    brill_freq_min_step: float
    wl_p: float
    wl_s: float
    yenista_filter_loss_p: float
    pump_attenuation_before_PD: float
    seg_len_m: float = field(init=False)
    stepsize_m: float = field(init=False)
    meters_scanned: float = field(init=False)
    brill_pump_start_idxs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seg_len_m = calc_segment_len_m(self.pulse_gen_freq)
        self.brill_pump_len_m = calc_pulse_len(
            self.pulse_gen_freq, self.brill_pump_seg_len
        )
        self.pump_len_m = calc_pulse_len(self.pulse_gen_freq, self.pump_seg_len)
        self.stepsize_m = self.seg_len_m * self.stepsize_seg
        self.meters_scanned = np.abs(self.start_idx - self.end_idx) * self.stepsize_m
        self.brill_pump_start_idxs = np.arange(
            self.start_idx, self.end_idx + self.stepsize_seg * 0.1, self.stepsize_seg
        )


@dataclass
class InstrumentSetupScope:
    brill_pump_seg_lens: list[int]
    brill_pump_module_idxs: list[int]
    seg_dist_between_pump_modules: int
    full_sequence_len: int
    pm_scope: Agilent3000X
    scope_avg_num: int
    brill_scope_ch: int
    fwm_scope_ch: int
    brill_scope_ch_dark_current_volt: float
    fwm_scope_ch_dark_current_volt: float
    responsivity_fwm: float
    responsivity_pump: float
    rf_clock: HP8341
    pulse_gen: HP81200Client
    len_for_brill_opt: float
    loss_between_meas_and_coll: np.ndarray
    component_names_loss_between_meas_and_coll: list
    diode_controller: EtekMLDC1032 | DiodeControllerRouter

    def __post_init__(self):
        self.pm_scope.average_count = self.scope_avg_num


@dataclass
class CollisionMeasurementDataScope:
    coords: np.ndarray
    coords_idxs: np.ndarray
    brill_powers: np.ndarray
    ref_brill_powers: np.ndarray
    opt_brill_freqs: np.ndarray
    pump_waveforms: list
    fwm_waveforms: list
    fwm_power: np.ndarray
    stepsize_m: float
    fut_IL: float
    fut_Ltot: float
    wl_p: float
    wl_s: float
    yenista_filter_loss_p: float
    pump_attenuation_before_PD: float
    brill_pump_pulse_len: float
    pump_len: float
    responsivity_fwm: float
    responsivity_pump: float
    loss_between_meas_and_coll: np.ndarray
    component_names_loss_between_meas_and_coll: list

    @classmethod
    def initialize(
        cls,
        sweep: MovingCollisionInputParamsScope,
        instr: InstrumentSetupScope,
        fut: Fiber,
    ):
        n = len(sweep.brill_pump_start_idxs)
        return cls(
            coords=(sweep.brill_pump_start_idxs - sweep.brill_pump_start_idxs[-1])
            * sweep.stepsize_m,
            coords_idxs=sweep.brill_pump_start_idxs,
            brill_powers=np.zeros(n),
            ref_brill_powers=np.zeros(n),
            opt_brill_freqs=np.zeros(n),
            pump_waveforms=[],
            fwm_waveforms=[],
            fwm_power=np.zeros(n),
            stepsize_m=sweep.stepsize_m,
            fut_IL=fut.insertion_loss,
            fut_Ltot=fut.length,
            wl_p=sweep.wl_p,
            wl_s=sweep.wl_p,
            yenista_filter_loss_p=sweep.yenista_filter_loss_p,
            pump_attenuation_before_PD=sweep.pump_attenuation_before_PD,
            brill_pump_pulse_len=sweep.brill_pump_len_m,
            pump_len=sweep.pump_len_m,
            responsivity_fwm=instr.responsivity_fwm,
            responsivity_pump=instr.responsivity_pump,
            loss_between_meas_and_coll=instr.loss_between_meas_and_coll,
            component_names_loss_between_meas_and_coll=instr.component_names_loss_between_meas_and_coll,
        )


def run_collision_sweep_scope(
    sweep_params: MovingCollisionInputParamsScope,
    instr: InstrumentSetupScope,
    data: CollisionMeasurementDataScope,
    filename: str,
    save_data: bool,
    num_segments_for_pulsegen: int = 5,
) -> CollisionMeasurementDataScope:
    idx_before_brill_opt = int(instr.len_for_brill_opt / sweep_params.seg_len_m)
    start_idxs = sweep_params.brill_pump_start_idxs
    cur_brill_freq = instr.rf_clock.frequency
    assert isinstance(cur_brill_freq, float)
    opt_brill_freq = cur_brill_freq
    t0 = time.time()
    for idx, pump_pulse_start_idx in enumerate(start_idxs):
        seq1 = np.arange(
            pump_pulse_start_idx, pump_pulse_start_idx + instr.brill_pump_seg_lens[0]
        )
        seq2_start_idx = pump_pulse_start_idx + instr.seg_dist_between_pump_modules
        seq2 = np.arange(seq2_start_idx, seq2_start_idx + instr.brill_pump_seg_lens[1])
        set_multiple_seqs(
            instr.pulse_gen,
            instr.full_sequence_len,
            [seq1, seq2],
            enable_output_after_change=False,
            module_ids=instr.brill_pump_module_idxs,
            num_segments=num_segments_for_pulsegen,
        )
        time.sleep(0.2)
        if idx % idx_before_brill_opt == 0 and idx != 0:
            # opt_brill_freq = optimize_brill_freq_golden_scope(
            #     instr.pm_scope,
            #     instr.brill_scope_ch,
            #     instr.rf_clock,
            #     cur_brill_freq - sweep_params.brill_freq_init_step,
            #     cur_brill_freq + sweep_params.brill_freq_init_step,
            #     sweep_params.brill_freq_min_step,
            # )
            opt_brill_freq = optimize_brill_freq_hill_climb_scope(
                instr.pm_scope,
                instr.brill_scope_ch,
                instr.rf_clock,
                initial_step=sweep_params.brill_freq_init_step,
                min_step=1e6,
            )
        cur_brill_freq = opt_brill_freq
        # TODO: THIS IS CURRENTLY JUST IN VOLTS SHOULD BE MADE INTO OPTICAL POWER AT SOME POINT
        instr.pm_scope.waveform.channel = instr.brill_scope_ch
        x_brill, y_brill = instr.pm_scope.waveform.read_waveform()
        y_brill = fix_discontinuity(y_brill)
        y_brill = y_brill - instr.brill_scope_ch_dark_current_volt
        brill_power = pulse_area(x_brill, y_brill)
        instr.pm_scope.waveform.channel = instr.fwm_scope_ch
        x_fwm, y_fwm = instr.pm_scope.waveform.read_waveform()
        y_fwm = fix_discontinuity(y_fwm)
        y_fwm = y_fwm - instr.fwm_scope_ch_dark_current_volt
        pump_waveform = np.vstack((x_brill, y_brill))
        fwm_waveform = np.vstack((x_fwm, y_fwm))
        fwm_power = pulse_area(x_fwm, y_fwm)
        data.brill_powers[idx] = brill_power
        data.opt_brill_freqs[idx] = opt_brill_freq
        data.fwm_power[idx] = fwm_power
        data.pump_waveforms.append(pump_waveform)
        data.fwm_waveforms.append(fwm_waveform)
        print_time_status(idx, t0, len(start_idxs))
        # save data every 10 measurement
        if idx % 10 == 0:
            if save_data:
                with open(f"{filename}", "wb") as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # obtain reference brill powers for each frequency used by looping through it with pump turned off
    # print("Now obtaining reference pump powers")
    # instr.diode_controller.disable_edfa("EDFA1")
    # t0 = time.time()
    # instr.pm_scope.waveform.channel = instr.brill_scope_ch
    # for idx, rf_freq in enumerate(data.opt_brill_freqs):
    #     instr.rf_clock.frequency = rf_freq
    #     x_brill, y_brill = instr.pm_scope.waveform.read_waveform()
    #     y_brill = fix_discontinuity(y_brill)
    #     y_brill = y_brill - instr.brill_scope_ch_dark_current_volt
    #     ref_brill_power = pulse_area(x_brill, y_brill)
    #     data.ref_brill_powers[idx] = ref_brill_power
    #     print(f"Done with {idx + 1} out of {len(data.opt_brill_freqs)}")
    # print(f"It took {(time.time() - t0):.1f}s")
    if save_data:
        with open(f"{filename}", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def full_filepath_collision_meas_scope(
    fiber: Fiber,
    input_params: MovingCollisionInputParamsScope,
    base_path: str = "/home/thjalfe/Documents/PhD/Projects/Experiments/external_stay/data/moving_collisions",
    extra_path: str = "",
    extra_filename: str = "",
):
    fiber_str = f"{fiber.full_fiber_name}"
    if len(extra_path) > 0:
        base_path = f"{base_path}/{extra_path}"
    dir = f"{base_path}/{fiber_str}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    base_filename = (
        f"brill-pump-dur={input_params.brill_pump_len_m:.1f}m_"
        f"m-scanned={input_params.meters_scanned:.1f}m_"
        f"wl-p={input_params.wl_p:.1f}nm_wl-s={input_params.wl_s:.1f}nm_"
        f"stepsize-m={input_params.stepsize_m:.1f}m{extra_filename}.pkl"
    )
    filepath = os.path.join(dir, base_filename)

    if not os.path.exists(filepath):
        return filepath

    # Append _n if file exists
    n = 1
    while True:
        alt_filename = base_filename.replace(".pkl", f"_{n}.pkl")
        alt_filepath = os.path.join(dir, alt_filename)
        if not os.path.exists(alt_filepath):
            return alt_filepath
        n += 1


fiber_dict = load_fibers_from_json()
fiber = get_fiber("fiber_2", fiber_dict)
pulselen_1 = 15
pulselen_4 = pulselen_1 + 2
pump_pulse_dur = calc_pulse_len(pulse_gen_freq, pulselen_1)
save_data = True
start_idx = 2150
start_idx = 2168
end_idx = start_idx - 600
# end_idx = start_idx - 350
brill_freq_init_step = 10e6
brill_freq_min_step = 1e6
osa.resolution = 0.05
# osa.sensitivity = "SHI3"
stepsize = -1
pump_pulse_modules = [1, 4]
meters_scanned = (start_idx - end_idx) * seg_len_m
stepsize_m = np.abs(stepsize * seg_len_m)
input_params = MovingCollisionInputParamsScope(
    pulse_gen_freq,
    pulselen_1,
    pulselen_2,
    stepsize,
    start_idx,
    end_idx,
    brill_freq_init_step,
    brill_freq_min_step,
    tls2.wavelength,
    tls1.wavelength,
    34.8 - 27.8,
    20,
)

dark_current_volts = get_dark_currents(
    [1, 2], small_scope, [diode_controller_1, diode_controller_2]
)
instrument_setup = InstrumentSetupScope(
    [pulselen_1, pulselen_4],
    [1, 4],
    idx_from_1_to_4,
    seq_len,
    small_scope,
    16000,
    1,
    2,
    dark_current_volts[0],
    dark_current_volts[1],
    0.92,
    0.95,
    rf_clock,
    hp81200_base,
    2,
    np.array([[20, 7, 6.4, 1.2, 0.5], [0, 1.9, 6.4, 1.2, 0.5]]),
    [
        [
            "man_attenuator",
            "yenista_filter",
            "4-way_coupler",
            "circulator",
            "HNLF_couple",
        ],
        [
            "Nothing",
            "KK-filters",
            "4-way_coupler",
            "circulator",
            "HNLF_couple",
        ],
    ],
    diode_controller,
)

data = CollisionMeasurementDataScope.initialize(
    input_params,
    instrument_setup,
    fiber,
)
filename = full_filepath_collision_meas_scope(
    fiber,
    input_params,
    extra_path="scope_pm/fully-from-end-fiber",
    extra_filename="",
)
start_scrambler(dc_1, voltage_stepsize=1)
data = run_collision_sweep_scope(
    input_params,
    instrument_setup,
    data,
    filename,
    True,
)
# |%%--%%| <XHamkD9JgF|wlZtbgmB4y>

fiber_dict = load_fibers_from_json()
fiber = get_fiber("fiber_2", fiber_dict)
pulselen_1 = 10
pulselen_4 = pulselen_1 + 2
pump_pulse_dur = calc_pulse_len(pulse_gen_freq, pulselen_1)
save_data = True
start_idx = 2165
end_idx = start_idx - 700
end_idx = start_idx - 250
brill_freq_init_step = 10e6
brill_freq_min_step = 1e6
osa.resolution = 0.05
osa.stop_sweep()
# osa.sensitivity = "SHI3"
stepsize = -1
pump_pulse_modules = [1, 4]
meters_scanned = (start_idx - end_idx) * seg_len_m
stepsize_m = np.abs(stepsize * seg_len_m)
pm_avg_time = 0.5
input_params = MovingCollisionInputParams(
    pulse_gen_freq,
    pulselen_1,
    pulselen_2,
    stepsize,
    start_idx,
    end_idx,
    brill_freq_init_step,
    brill_freq_min_step,
    tls2.wavelength,
    tls1.wavelength,
    34.8 - 27.8,
    33 - 26,
)
instrument_setup = InstrumentSetup(
    [pulselen_1, pulselen_4],
    [1, 4],
    idx_from_1_to_4,
    seq_len,
    pm_1,
    pm_2,
    rf_clock,
    hp81200_base,
    diode_controller,
    pm_avg_time,
)

data = CollisionMeasurementData.initialize(
    input_params,
    instrument_setup,
    fiber,
)
filename = full_filepath_collision_meas(fiber, input_params, extra_path="scope_pm")
diode_controller.set_edfa_to_max("EDFA1")
start_scrambler(dc_1, voltage_stepsize=1)
data = run_collision_sweep(
    input_params,
    instrument_setup,
    data,
    filename,
    True,
)
# |%%--%%| <wlZtbgmB4y|m5kQzJzaXh>
osa.zero_nm_sweeptime = 5
osa.sweeptype = "SGL"
osa.sensitivity = "SHI3"
osa.sweep()
x = get_osa_spec(osa)
print(np.mean(x[1]))
osa.sweeptype = "RPT"
osa.sweep()
# |%%--%%| <m5kQzJzaXh|TPaZZ1hIJh>
plt.hist(x[1])
plt.show()
