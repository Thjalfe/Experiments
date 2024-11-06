# %%
import pickle
import numpy as np
from typing import List
import os
import time
import matplotlib.pyplot as plt
import sys
import importlib
import shutil


sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser, Laser, TiSapphire
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from toptica_control import DLCcontrol
from misc import PM
from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons
from run_sweep_funcs import (
    sweep_both_pumps,
    calculate_approx_idler_loc,
    find_sig_idler_powers,
    optimize_one_pump_wl,
    optimize_one_pump_wl_increasing_fineness,
    increment_pump_wl,
    run_pol_opt,
)

# from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons
from arduino_pm import ArduinoADC

importlib.reload(sys.modules["run_sweep_funcs"])

plt.ion()
import pyvisa as visa


def manual_pol_opt(
    osa,
    idler_wl,
    idler_power,
    level_scale=2,
    num_divisions_below_start_power=2,
    pol_opt_method="manual",
    pm=None,
    pol_con1=None,
    pol_con2=None,
):
    old_span = osa.span
    if pol_opt_method == "manual":
        run_pol_opt(
            osa, idler_wl, idler_power, level_scale, num_divisions_below_start_power
        )
    elif pol_opt_method == "auto":
        run_pol_opt(
            osa,
            idler_wl,
            idler_power,
            level_scale,
            num_divisions_below_start_power,
            pol_opt_method="auto",
            pm=pm,
            pol_con1=pol_con1,
            pol_con2=pol_con2,
        )
    osa.span = old_span


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1595
pump2_start = 1578

pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.05)
ando_pow_start = 0
arduino = ArduinoADC("COM10")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pump1_laser = AndoLaser(pump1_start, GPIB_address=24, power=8)
pump2_laser = AndoLaser(pump2_start, GPIB_address=23, power=8)
pump1_laser.linewidth = 1
pump2_laser.linewidth = 1
time.sleep(0.1)
pump1_laser.enable()
pump2_laser.enable()
# sig_laser = DLCcontrol(ip="192.168.1.201")
# sig_laser.wavelength = 971.215
sig_laser = TiSapphire(5)
verdi = VerdiLaser(12)
# verdi.power = 7
osa = OSA(
    960,
    990,
    resolution=0.1,
    GPIB_address=19,
    sweeptype="RPT",
)
osa.level = -10
# scope = TektronixOscilloscope()
time.sleep(0.5)
try:
    thorlabs_pm = PM()
except AttributeError:
    thorlabs_pm = None
    print("Thorlabs PM not connected")

# %% Manually moving pump wavelength, const tisa
### saving spectra
upper_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\TMSI_tisa=965.1_pump_L_band_sweep\manual_precise_sweep"
tisa_loc = 965.1
dc = 0.04
pico.awg.set_square_wave_duty_cycle(pulse_freq, dc)
pump1_laser.wavelength = 1610

# %%
dir_name = os.path.join(upper_dir, str(dc))
filename = f"spec_{pump1_laser.wavelength:.3f}_{pump2_laser.wavelength:.3f}.pkl"
num_reps = 5
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
old_res = osa.resolution
osa.resolution = 0.05
osa.stop_sweep()
osa.sweeptype = "SGL"
spectra = []
for i in range(num_reps):
    osa.sweep()
    wavelengths = osa.wavelengths
    powers = osa.powers
    spectrum = np.vstack((wavelengths, powers))
    spectra.append(spectrum)
spectra = np.array(spectra)
with open(os.path.join(dir_name, filename), "wb") as file:
    pickle.dump(spectra, file)
osa.resolution = old_res
osa.sweeptype = "RPT"
osa.sweep()

# %%
sig_loc_guess = 971.4
sig_laser.set_wavelength_iterative_method(sig_loc_guess, osa)
osa.span = (sig_loc_guess - 5, sig_loc_guess + 5)
osa.stop_sweep()
osa.sweeptype = "SGL"
osa.sweep()
sig_wl = osa.wavelengths[np.argmax(osa.powers)]
print(f"Signal wavelength: {sig_wl:.1f}")
pump1_wl = 1591.5
pump1_laser.wavelength = pump1_wl
pump1_start = 1590.75
pump1_stop = 1593.05
# pump1_start = 1593.75
# pump1_stop = 1595.75
osa.level = -10
pump1_stepsize = 0.5
pump1_array = np.arange(pump1_start, pump1_stop, pump1_stepsize)
if len(pump1_array) == 0 or pump1_array[-1] != pump1_stop:
    pump1_array = np.append(pump1_array, pump1_stop)
pump2_wl_start = 1605  # If an educated guess for the first pump2 wl is available
pump2_wl_stop = 1611
pump2_wl_min = 1594.5
pump2_wl_max = 1611
# pump2_wl_start = 1588  # If an educated guess for the first pump2 wl is available
# pump2_wl_stop = 1591.7
# pump2_wl_min = 1570
# pump2_wl_max = 1591.7
pump2_wl_move_factor = (
    10  # how much to move pump2 wl away from its prev optimum based on pump1 stepsize
)
pump2_stepsize = 0.1
min_distance_between_pumps_nm = 2
stepsize = 0.1
finer_p2_stepsize = [0.025]
# stepsizes = [0.2, 0.1, 0.05]
idler_tolerance_nm = 0.4
osa.resolution = 0.1
old_osa_res = osa.resolution
data_loc = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl={sig_wl:.1f}nm\sweep_both_pumps"
data_loc = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl_same-as-qd\sweep_both_pumps"
data_loc = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl_same-as-qd\sweep_both_pumps_auto_pol_opt_780-fiber-out"
merge_data = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl_same-as-qd\sweep_both_pumps\p1_wl=1594.8-1595.3_p2_wl=1577.4-1586.0_p1_stepsize=0.50_p2_stepsize=0.20\data.pkl"
merge_data = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl_same-as-qd\sweep_both_pumps_auto_pol_opt_780-fiber-out\p1_wl=1590.5-1591.5_p2_wl=1595.8-1611.0_p1_stepsize=1_p2_stepsize=0.1\data.pkl"
merge_data = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl_same-as-qd\sweep_both_pumps_auto_pol_opt_780-fiber-out\p1_wl=1593.8-1595.8_p2_wl=1575.6-1591.7_p1_stepsize=0.50_p2_stepsize=0.10\data.pkl"
dc_array = np.array([0.05, 0.1, 0.125, 0.2, 0.25, 0.5, 1])


data_dict = sweep_both_pumps(
    pump1_laser,
    pump2_laser,
    None,
    osa,
    pump1_array,
    pump2_wl_min,
    pump2_wl_max,
    pump2_wl_stop,
    pump1_stepsize,
    pump2_stepsize,
    sig_wl,
    min_distance_between_pumps_nm,
    pico.awg,
    osa_res=0.1,
    tolerance_nm=0.5,
    plot_ce_progression=True,
    save_data=True,
    data_loc=data_loc,
    plot_sub_ce_progression=False,
    save_sub_data=False,
    data_sub_loc=None,
    dc_array=dc_array,
    increased_fineness_stepsizes=finer_p2_stepsize,
    num_prev_res_to_cover=2,
    pump2_wl_start=pump2_wl_start,
    pump2_wl_move_factor=pump2_wl_move_factor,
    sweep_full_p2_range=False,
    verdi=None,
    turn_off_after_sweep=False,
    merge_data=None,
    pol_opt_method="auto",
    pm=arduino,
    pol_con1=pol_con1,
    pol_con2=pol_con2,
)
# optimize_one_pump_wl_increasing_fineness(
#     pump2_laser,
#     pump1_wl,
#     tisa_loc,
#     pump2_wl_start,
#     pump2_wl_stop,
#     stepsizes,
#     osa_res=0.1,
#     tolerance_nm=idler_tolerance_nm,
#     plot_ce_progression=True,
#     save_data=True,
#     data_loc=data_loc,
#     dc=0.05,
# )
# pump2_wl_start = 1588
# pump2_wl_stop = 1589
osa.sweeptype = "RPT"
osa.sweep()
# %%
data_loc = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl_same-as-qd\sweep_both_pumps\p1_wl=1594.3-1595.3_p2_wl=1577.1-1589.0_p1_stepsize=0.50_p2_stepsize=0.20\data.pkl"
data_dict = pickle.load(open(data_loc, "rb"))


def pump_wls_to_thz_sep(p1wl, p2wl, c=299792458):
    p1wl_thz = c / (p1wl * 10**-9) * 10**-12
    p2wl_thz = c / (p2wl * 10**-9) * 10**-12
    return np.abs(p1wl_thz - p2wl_thz)


pump_sep_thz = []
ce_pol_opt_peak_tmp = []
for p1_tmp in data_dict.keys():
    if type(p1_tmp) is not np.float64:
        continue
    if "ce_peak_pol_opt" not in data_dict[p1_tmp].keys():
        continue
    print(p1_tmp)
    pump_sep_thz.append(
        pump_wls_to_thz_sep(
            p1_tmp,
            data_dict[p1_tmp]["p2_max_ce"],
        ),
    )
    print(np.mean(data_dict[p1_tmp]["ce_peak_pol_opt"][0]))
    ce_pol_opt_peak_tmp.append(np.mean(data_dict[p1_tmp]["ce_peak_pol_opt"][0]))
print(f"Separation so far: {np.round(pump_sep_thz,3)} THz")
print(f"CE peak so far: {np.round(ce_pol_opt_peak_tmp, 3)} dB")
# %%  Sweep one wl while keeping the other constant
pump1_wl = 1592.75
pump1_wl = 1594.5
pump1_laser.wavelength = pump1_wl
sig_wl = 971.4
sig_laser.set_wavelength_iterative_method(sig_wl, osa)
data_loc = rf"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\QD\data\classical-opt\sig-wl={sig_wl:.1f}nm\opt-one-pump_other_const={pump1_wl:.1f}nm"
pump2_wl_start = 1583.7
pump2_wl_stop = 1584.1
# pump2_wl_start = 1609
# pump2_wl_stop = 1611
stepsize = 0.05
# pump2_wl_stop = pump1_wl - 2
idler_tolerance_nm = 0.5
dc = 0.05
pico.awg.set_square_wave_duty_cycle(pulse_freq, dc)
ce_array, spectra, pump2_max_ce = optimize_one_pump_wl(
    pump2_laser,
    osa,
    pump1_wl,
    sig_wl,
    pump2_wl_start,
    pump2_wl_stop,
    stepsize,
    idler_tolerance_nm=idler_tolerance_nm,
    plot_ce_progression=True,
    save_data=False,
    data_loc=data_loc,
    dc=dc,
    idler_side="red",
)

osa.sweeptype = "RPT"
osa.sweep()
# %%
manual_pol_opt(
    osa,
    975.37,
    -20,
    pol_opt_method="auto",
    pm=arduino,
    pol_con1=pol_con1,
    pol_con2=pol_con2,
)
