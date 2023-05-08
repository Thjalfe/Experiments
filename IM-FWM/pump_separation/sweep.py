
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
GPIB_val = 0
# |%%--%%| <nhkI9B4MS1|XNRNgeMOZV>
TiSa = TiSapphire(3)
ando1_start = 1572
ando2_start = 1570
ando1 = laser("ando", ando1_start, power=8, GPIB_num=GPIB_val)
ando2 = laser("ando2", ando2_start, power=8, GPIB_num=GPIB_val)
osa = OSA(
    ando2_start - 2,
    ando1_start + 2,
    resolution=0.05,
    GPIB_num=[GPIB_val, 19],
    sweeptype="SGL",
)
# osa2 = OSA(ando2_start - 2, ando1_start + 2, resolution=0.05, GPIB_num=[GPIB_val, 18], sweeptype="SGL")

# |%%--%%| <KmQalyAPfu|vD92Ji6HUF>
wl_tot = 2
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
data_folder = ["./data/pulse/3us_25duty/sortby_redshift/"]
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
        adjust_laser_wavelengths=True,
        equal_pump_power=equal_pump_power,
        log_pm=log_pm,
        OSA_sens="SHI1",
        OSA_GPIB_num=[0, 19],
        sortpeaksby="red",
    )
# |%%--%%| <3B182la2sh|4RG6bjG2Rz>
# Check pumps
osa.set_res(0.1)
for i in range(len(ando1_wl)):
    ando1.set_wavelength(ando1_wl[i])
    ando1.adjust_wavelength(OSA_GPIB_num=[GPIB_val, 18])
    ando2.set_wavelength(ando2_wl[i])
    ando2.adjust_wavelength(OSA_GPIB_num=[GPIB_val, 18])
    osa2.set_span(ando2_wl[i] - 2, ando1_wl[i] + 2)
    osa2.set_res(0.1)
    osa2.sweep()
    osa2.save(f"./data/char_setup/pump_powers/wdm_tap_{ando2_wl[i]}_{ando1_wl[i]}")
    osa.set_span(ando2_wl[i] - 2, ando1_wl[i] + 2)
    osa.sweep()
    osa.save(f"./data/char_setup/pump_powers/throughsetup_{ando2_wl[i]}_{ando1_wl[i]}")
#|%%--%%| <4RG6bjG2Rz|xp8WI9xiP7>
from funcs.experiment import save_sweep
data_folder = "./data/pulse/3us_50duty/manual_specs/"
idx = 10
wl1_temp = ando1_wl[idx]
wl2_temp = ando2_wl[idx]
ando1.set_wavelength(wl1_temp)
ando2.set_wavelength(wl2_temp)
ando1.adjust_wavelength(OSA_GPIB_num=[GPIB_val, 19])
ando2.adjust_wavelength(OSA_GPIB_num=[GPIB_val, 19])
OSA_temp = OSA(
    981,
    984,
    resolution=0.05,
    sensitivity='SMID',
    GPIB_num=[GPIB_val, 19],
)
OSA_temp.set_sweeptype('RPT')
#|%%--%%| <xp8WI9xiP7|PFwTvIon5C>
OSA_temp.set_sens('SHI1')
OSA_temp.sweep()
OSA_temp.get_spectrum()
save_sweep(
    data_folder,
    f"{wl2_temp}_{wl1_temp}_0",
    wavelengths=OSA_temp.wavelengths,
    powers=OSA_temp.powers,
    ando_powers=(ando1.power, ando2.power),
)
