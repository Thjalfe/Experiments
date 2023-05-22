import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from funcs.experiment import even_ando_power

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
GPIB_val = 1
#|%%--%%| <MUNfIyzM17|0WxXXHEfw1>
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
osa = OSA(
    ando2_start - 2,
    ando1_start + 2,
    resolution=0.05,
    GPIB_num=[GPIB_val, 19],
    sweeptype="SGL",
)

#|%%--%%| <StXATCf7na|MUNfIyzM17>
save_dir = "./data/calibration/edfa_power_calibration/"
ando2.laserOFF()
wls = np.arange(1565, 1595)
edfa_power_cal = np.zeros((2, len(wls)))
target_power = 0
for idx, wl in enumerate(wls):
    ando1.set_power(target_power)
    ando1.set_wavelength(wl)
    time.sleep(0.1)
    cur_p_in = ando1.get_true_power()
    if cur_p_in < target_power:
        ando1.set_power(target_power + np.abs(cur_p_in - target_power))
    else:
        ando1.set_power(target_power - np.abs(cur_p_in - target_power))
    osa.set_span(wl - 0.25, wl + 0.25)
    osa.sweep()
    print(np.max(osa.powers))
    edfa_power_cal[0, idx] = wl
    edfa_power_cal[1, idx] = np.max(osa.powers)
#|%%--%%| <0WxXXHEfw1|1sNHZey8iR>
from scipy.interpolate import CubicSpline
# def even_edfa_power(ando1, ando2, power_data):
#     power_spline = CubicSpline(power_data[0], power_data[1]))
#     power_diff = power_spline(ando1.actual_wavelength) - power_spline(ando2.actual_wavelength)
#     if power_diff > 0:


# |%%--%%| <tt2FMhe5vb|vtAVRl7ssV>
# even out laser power output
from instrument_class import PM

pm = PM()
save_dir = "./data/calibration/ando_power_calibration/"
wl_power_calibration = np.arange(1520, 1600.5, 0.5)
ando_pows = np.arange(-7, 8)
power_dict = {
    "wl": wl_power_calibration,
    "power_settings": ando_pows,
    "powers": np.zeros((2, len(ando_pows), len(wl_power_calibration))),
    "GPIB_adresses": [24, 23],
}
#|%%--%%| <vtAVRl7ssV|ZpXzc0pC30>
for idx_pow, ando_pow in enumerate(ando_pows):
    for idx_wl, wl in enumerate(wl_power_calibration):
        ando1.set_power(ando_pow)
        time.sleep(0.1)
        ando1.set_wavelength(wl)
        time.sleep(0.5)
        power_dict["powers"][0, idx_pow, idx_wl] = pm.read()
#|%%--%%| <ZpXzc0pC30|6xK207Ne0s>
for idx_pow, ando_pow in enumerate(ando_pows):
    for idx_wl, wl in enumerate(wl_power_calibration):
        ando2.set_wavelength(wl)
        time.sleep(0.5)
        power_dict["powers"][1, idx_pow, idx_wl] = pm.read()
# save data
import pickle

with open(save_dir + "both_0_dBm.pkl", "wb") as f:
    pickle.dump(power_dict, f)
# |%%--%%| <s7zUJToua8|xtvCXpbV0l>
# plot data
plt.plot(power_dict["wl"], power_dict["powers"][0], label="ando1")
plt.plot(power_dict["wl"], power_dict["powers"][1], label="ando2")
plt.legend()
plt.show()
#|%%--%%| <xtvCXpbV0l|rp87px8KZv>
save_dir = "./data/calibration/edfa_power_calibration/"
data_folder = ["./data/pump_DFWM/even_pump_p/const_pump_sep", "./data/pump_DFWM/even_pump_p/const_mean_pump_wl"]
ando2.laserOFF()
wls = np.arange(1565, 1595)
edfa_power_cal = np.zeros((2, len(wls)))
target_power = 0
for idx, wl in enumerate(wls):
    ando1.set_power(target_power)
    ando1.set_wavelength(wl)
    time.sleep(0.1)
    cur_p_in = ando1.get_true_power()
    if cur_p_in < target_power:
        ando1.set_power(target_power + np.abs(cur_p_in - target_power))
    else:
        ando1.set_power(target_power - np.abs(cur_p_in - target_power))
    print("Power before adjustment: ", cur_p_in)
    print("Power after adjustment: ", ando1.get_true_power())
    osa.set_span(wl - 0.25, wl + 0.25)
    osa.sweep()
    print(np.max(osa.powers))
    edfa_power_cal[0, idx] = wl
    edfa_power_cal[1, idx] = np.max(osa.powers)
