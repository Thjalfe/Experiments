import numpy as np
import matplotlib.pyplot as plt
import sys
from funcs.experiment import run_experiment, new_sig_start

# from util_funcs import load_raw_data
sys.path.append("U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl")
from OSA_control import OSA
from laser_control import laser, TiSapphire


plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
GPIB_val = 1
# |%%--%%| <nhkI9B4MS1|9Am64xPXlA>
TiSa = TiSapphire(3)
ando1_start = 1572
ando2_start = 1570
ando1 = laser("ando", ando1_start, power=0, GPIB_num=GPIB_val)
ando2 = laser("ando2", ando2_start, power=0, GPIB_num=GPIB_val)
osa = OSA(
    ando2_start - 2,
    ando1_start + 2,
    resolution=0.05,
    GPIB_num=[GPIB_val, 19],
    sweeptype="SGL",
)
# osa2 = OSA(ando2_start - 2, ando1_start + 2, resolution=0.05, GPIB_num=[GPIB_val, 18], sweeptype="SGL")

# |%%--%%| <TcHuziQRWR|vD92Ji6HUF>
wl_tot = 1.2
del_wl = 0.1
num_sweeps = int(wl_tot / del_wl)
ando1_wl = np.arange(1571.5, 1577.5, 0.5)
ando2_wl = np.arange(1570.5, 1564.5, -0.5)
sig_start = 981.5
full = np.append(ando1_wl, ando2_wl)
if not len(ando2_wl) == len(ando1_wl):
    raise IndexError(
        f"ando1 and ando2 wl arrays must be the same length, currently they have lengths {len(ando1_wl)} and {len(ando2_wl)} respectively"
    )
# |%%--%%| <5kdYCanZM7|3B182la2sh>
equal_pump_power = False
log_pm = False
# data_folder = ["./data/pulse/3us_25duty/sortby_redshift/"]
data_folder = ["./data/CW/2.15W/"]
data_folder = ["./data/CW/2W/"]
for i in range(len(data_folder)):
    TiSa.set_wavelength(sig_start, OSA_GPIB_num=[GPIB_val, 19])
    run_experiment(
        data_folder[i],
        ando1,
        ando2,
        TiSa,
        ando1_wl,
        ando2_wl,
        num_sweeps,
        del_wl,
        wl_tot,
        sig_start,
        adjust_laser_wavelengths=False,
        equal_pump_power=equal_pump_power,
        log_pm=log_pm,
        OSA_sens="SHI1",
        OSA_GPIB_num=[GPIB_val, 19],
        sortpeaksby="red",
    )
# |%%--%%| <3B182la2sh|4RG6bjG2Rz>
# redo measurements
data_folder = ["./data/CW/2.15W/"]
pump_sep = np.array([7, 9])
mean_pump_wl = 1571


def redo_specific_pump_sep(pump_sep, mean_pump_wl, data_folder):
    ando1_wl = mean_pump_wl + pump_sep / 2
    ando2_wl = mean_pump_wl - pump_sep / 2
    sig_start = np.zeros(len(pump_sep))
    for i in range(len(pump_sep)):
        sig_start[i] = new_sig_start(
            data_folder, ando2_wl[i] + 0.5, ando1_wl[i] - 0.5, wl_tot, -40, "red"
        )
    return ando1_wl, ando2_wl, sig_start


ando1_wl, ando2_wl, sig_start = redo_specific_pump_sep(
    pump_sep, mean_pump_wl, data_folder[0]
)

TiSa.set_wavelength(sig_start[0], OSA_GPIB_num=[GPIB_val, 19])
run_experiment(
    data_folder[0],
    ando1,
    ando2,
    TiSa,
    ando1_wl,
    ando2_wl,
    num_sweeps,
    del_wl,
    wl_tot,
    sig_start,
    adjust_laser_wavelengths=True,
    equal_pump_power=False,
    log_pm=False,
    OSA_sens="SHI1",
    OSA_GPIB_num=[GPIB_val, 19],
    sortpeaksby="red",
)
# |%%--%%| <4RG6bjG2Rz|JealLFJzb6>
from funcs.experiment import save_sweep

idx = 0
wl1_temp = ando1_wl[idx]
wl2_temp = ando2_wl[idx]
ando1.set_wavelength(wl1_temp)
ando2.set_wavelength(wl2_temp)
# ando1.adjust_wavelength(OSA_GPIB_num=[GPIB_val, 19])
# ando2.adjust_wavelength(OSA_GPIB_num=[GPIB_val, 19])
# OSA_temp = OSA(
#     981,
#     984,
#     resolution=0.05,
#     sensitivity='SMID',
#     GPIB_num=[GPIB_val, 19],
# )
# OSA_temp.set_sweeptype('RPT')
# |%%--%%| <JealLFJzb6|n0Bsq9IGYE>
osa.get_spectrum()
file_name = "./data/michael/idler_CE_1e5Hz_50duty"
# plt.plot(osa.wavelengths, osa.powers)
osa.save(file_name)
