import numpy as np
import matplotlib.pyplot as plt
import sys
from experiment_funcs import run_experiment
from util_funcs import load_raw_data
sys.path.append("U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl")
from OSA_control import OSA
from laser_control import laser, TiSapphire


plt.ion()
# import pyvisa as visa

# rm = visa.ResourceManager()
# print(rm.list_resources())
GPIB_val = 0
# |%%--%%| <nhkI9B4MS1|BLJsMAQTCU>
TiSa = TiSapphire(3)
ando1 = laser("ando", 1572, power=8, GPIB_num=GPIB_val)
ando2 = laser("ando2", 1570, power=8, GPIB_num=GPIB_val)
osa = OSA(
    979, 984, resolution=0.05, GPIB_num=[GPIB_val, 19], sweeptype="SGL"
)

# |%%--%%| <BLJsMAQTCU|vD92Ji6HUF>
wl_tot = 5
del_wl = 0.1
num_sweeps = int(wl_tot / del_wl)
ando1_wl = np.arange(1572, 1578.5, 0.5)
ando2_wl = np.arange(1570, 1563.5, -0.5)
sig_start = 980
full = np.append(ando1_wl, ando2_wl)
# |%%--%%| <5kdYCanZM7|3B182la2sh>


equal_pump_power = [False, True]
# equal_pump_power = [True]
log_pm = False
data_folder = ["./data_CW/", "./data_CW_equal_pumps/"]
data_folder = ["./data_pulsed/", "./data_pulsed_equal_pumps/"]
# data_folder = ['./data_CW_equal_pumps/']
for i in range(len(data_folder)):
    TiSa.set_wavelength(sig_start, OSA_GPIB_num=[GPIB_val, 19])
    run_experiment(
        data_folder[i],
        ando1_wl,
        ando2_wl,
        equal_pump_power[i],
        log_pm,
        num_sweeps,
        del_wl,
        wl_tot,
        sig_start,
        OSA_sens="SMID",
        OSA_GPIB_num=[0, 19],
    )
# |%%--%%| <3B182la2sh|hqn5mwNaCv>
pumpwl_low = 1570
pumpwl_high = 1572
data_folder = ["./data_CW_equal_pumps/"]
test = load_raw_data(data_folder[0], -80, pump_wl_pairs=[(pumpwl_low, pumpwl_high)])
# peaks_sorted = analyze_data(data_folder[0], pump_wl_pairs=[(pumpwl_low, pumpwl_high,)])[
#     (pumpwl_low, pumpwl_high)
# ]
# import pickle
# files = ['./data_CW_equal_pumps/1570.0_1572.0_0.pkl', './data_CW_equal_pumps/1570.0_1572.0_1.pkl']
# x = []
# for file in files:
#     with open(file, 'rb') as f:
#         x.append(pickle.load(f))
# |%%--%%| <hqn5mwNaCv|NMNquv6eW6>
