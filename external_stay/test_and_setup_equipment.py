import importlib
from typing import cast
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


from clients.dc_power_supply_clients import E3631A
from clients.diode_controller_clients import EtekMLDC1032
from clients.laser_clients import AgilentLaserClient
from clients.osa_clients import OSAClient
from clients.oscilloscope_clients import HP83480
from clients.power_meter_clients import Agilent8163x
from clients.rf_clock_clients import HP8341
from processing.helper_funcs import mW_to_dBm, dBm_to_mW
from run_exp_funcs.osnr import calc_OSNR, window_sidebands, osnr_multiple_laser_powers
from run_exp_funcs.bias import get_voltage_sign, sweep_all_dc_supplies, plot_mzm_biases

plt.style.use("custom")
plt.ioff()
# |%%--%%| <Ngf3b6rjw6|1gkR3aiNPX>
base_url = "http://100.80.229.245:5000"
wl1_init = 1550
wl2_init = 1550
tls1 = AgilentLaserClient(
    base_url,
    "agilent_laser_1",
    target_wavelength=wl1_init,
    power=0,
    source=0,
    GPIB_address=21,
)
tls2 = AgilentLaserClient(
    base_url,
    "agilent_laser_2",
    target_wavelength=wl2_init,
    power=0,
    source=2,
    GPIB_address=20,
)
tls1.wavelength = wl1_init
tls2.wavelength = wl2_init
tls1.enable()
tls2.enable()
pm_1 = Agilent8163x(base_url, "agilent_pm_1", 1550, channel=2, GPIB_address=20)
pm_2 = Agilent8163x(base_url, "agilent_pm_2", 1550, channel=3, GPIB_address=20)
pm_3 = Agilent8163x(base_url, "agilent_pm_3", 1550, channel=4, GPIB_address=20)
rf_clock = HP8341(base_url, "hp8341", frequency=1e9, power=0, GPIB_address=19)
scope = HP83480(base_url, "hp83480", "BYTE", GPIB_address=5)
dc_1 = E3631A(
    base_url,
    "e3631a_1",
    connected_channels=[2, 3],
    current_limit=0.01,
    limit_all_channels_on_init=True,
    GPIB_address=1,
)
dc_2 = E3631A(
    base_url,
    "e3631a_2",
    connected_channels=[2],
    current_limit=0.01,
    limit_all_channels_on_init=True,
    GPIB_address=2,
)
edfas_1 = {
    "EDFA2": [14, 21, 22],
    "EDFA3": [19, 20],
}
edfas_2 = {
    "EDFA1": [3, 5, 9, 13],
    "EDFA4": [1],
}
mA_lims_1 = {
    14: 500,
    19: 280,
    20: 280,
    21: 220,
    22: 220,
}

mA_lims_2 = {
    1: 400,
    3: 750,
    5: 300,
    9: 300,
    13: 750,
}
diode_controller_1 = EtekMLDC1032(
    base_url, "etek_mldc_1", edfas=edfas_1, mA_lims=mA_lims_1, GPIB_address=6
)
diode_controller_2 = EtekMLDC1032(
    base_url, "etek_mldc_2", edfas=edfas_2, mA_lims=mA_lims_2, GPIB_address=7
)
osa = OSAClient(base_url, "osa_1", 1550, GPIB_address=25, GPIB_bus=0)
# |%%--%%| <1gkR3aiNPX|AeOp1b8irn>
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

# |%%--%%| <BMM3Ne6C1P|J8mkkZB8AG>
# SCOPE MEASUREMENTS
scope_channel_idx = 0
scope_channels = [3, 4]
scope.waveform.channel = scope_channels[scope_channel_idx]
if scope.waveform.channel == 3:
    dc_channel = 3
elif scope.waveform.channel == 4:
    dc_channel = 2
else:
    raise ValueError("Invalid scope channel")
dc = dc_1
dc.cur_channel = dc_channel
flip_voltage = get_voltage_sign(dc)
# sweep_biases = np.arange(1, 2 + 0.1, 0.4) * flip_voltage
sweep_biases = np.arange(10, 12 + 0.1, 0.4) * flip_voltage
# sweep_biases = np.arange(0, 12 + 0.1, 1) * flip_voltage
x, y = [], []
for bias in sweep_biases:
    dc.voltage = bias
    time.sleep(0.1)
    x_tmp, y_tmp = scope.waveform.read_waveform()
    x.append(x_tmp)
    y.append(y_tmp)
# |%%--%%| <1zFJVeIRJN|Kqt6L30ugV>
# Investigate OSNR
data_dir = "data/OSNR_char/no_rf_in/P1-FWM-pump"
save_data = True
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
osa.resolution = 0.1  # Standard resolution for OSNR in optical comms
osa.span = (1530, 1570)  # C band
osa.level_scale = 10
osa.sweeptype = "SGL"
# loc_str = "loc4_pre-amp-at-loc1-and-loc3"
laser_powers = np.arange(0, 2 + 0.1, 0.5)
laser_wavelengths = [1550]
spectra_dict = {"spectra": [], "wavelengths": laser_wavelengths, "powers": laser_powers}
spectra_dict["spectra_rolled"] = []
spectra_dict["OSNR"] = []
dist_from_center = 0.5
window_size_nm = 1
wl = laser_wavelengths[0]
tls2.wavelength = wl
spectra_dict = osnr_multiple_laser_powers(
    laser_powers,
    tls2,
    osa,
    spectra_dict,
    dist_from_center,
    window_size_nm,
    save_data,
    data_dir,
    loc_str,
)
