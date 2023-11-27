# %%
import numpy as np
from typing import List
import os
import time
import matplotlib.pyplot as plt
import sys
import pickle


sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser, TiSapphire
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from amonics_edfa import EDFA
from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons, Agilent11896A
from funcs.sweep_tisa_around_opt import sweep_w_pol_opt_based_on_linear_fit
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA
from misc import PM

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
ando1_start = 1577
ando2_start = 1563
ipg_edfa = IPGEDFA(connection_mode="GPIB", GPIB_address=17)
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.5)
ando_pow_start = 0
ando1_laser = AndoLaser(ando1_start, GPIB_address=24, power=8)
ando2_laser = AndoLaser(ando2_start, GPIB_address=23, power=8)
time.sleep(0.1)
ando1_laser.laser_on()
ando2_laser.laser_on()
tisa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    960,
    990,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="RPT",
)
pm = PM()

# verdi.power = 6
# verdi.shutter = 0
time.sleep(0.5)
# verdi.shutter = 1
ipg_edfa.status = 1


# %% power meter stuff
ando1_wl_arr = np.arange(1570, 1609, 1)

ando2_wl_arr = np.arange(1563, 1531, -1)
duty_cycles = np.array([0.1, 0.2, 0.25, 0.5, 1])
data = {
    "L_band": {
        dc: np.column_stack((ando1_wl_arr, np.full(ando1_wl_arr.shape, np.nan)))
        for dc in duty_cycles
    },
    "C_band": {
        dc: np.column_stack((ando2_wl_arr, np.full(ando2_wl_arr.shape, np.nan)))
        for dc in duty_cycles
    },
}
for wl1_idx, wl1 in enumerate(ando1_wl_arr):
    for dc in duty_cycles:
        ando1_laser.wavelength = wl1
        pico.awg.set_square_wave_duty_cycle(pulse_freq, dc)
        time.sleep(3)
        data["L_band"][dc][wl1_idx, 1] = pm.read()

time.sleep(0.5)
ipg_edfa.status = 1
time.sleep(0.5)
ando1_laser.laser_off()
time.sleep(10)
# %%
for wl2_idx, wl2 in enumerate(ando2_wl_arr):
    for dc in duty_cycles:
        ando2_laser.wavelength = wl2
        pico.awg.set_square_wave_duty_cycle(pulse_freq, dc)
        time.sleep(3)
        data["C_band"][dc][wl2_idx, 1] = pm.read()

with open("./data/C_plus_L_band/cleo/pump_powers_after_FMF.pkl", "wb") as f:
    pickle.dump(data, f)

# %% Pump spectra
ando1_start = 1577
ando2_start = 1563
pump_stepsize = 2.5
total_pump_move_nm = 30
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1

pump_wl_list = [
    (ando1_start + pump_stepsize * i, ando2_start - pump_stepsize * i)
    for i in range(num_pump_steps)
]


# %% Some more general functions
def calculate_approx_idler_loc(tisa_wl: float, pump_wls: np.ndarray, idler_side: str):
    if idler_side == "red":
        return 1 / (1 / pump_wls[0] + 1 / tisa_wl - 1 / pump_wls[1])
    elif idler_side == "blue":
        return 1 / (1 / pump_wls[1] + 1 / tisa_wl - 1 / pump_wls[0])
    else:
        raise ValueError("idler_side must be either 'red' or 'blue'")


def optimize_pump_pols_at_tisa_wl(
    tisa_wl: float,
    pump_wls: np.ndarray,
    idler_side: str,
    tisa: TiSapphire,
    osa: OSA,
    arduino: ArduinoADC,
    *pol_cons: ThorlabsMPC320,
    max_or_min: str = "max",
    interval: float = 0.005,
    tolerance: float = 0.2,
):
    def check_idler_power(osa: OSA, min_idler_power: float = -50):
        """
        For the osa to give reliable power output to the arduino, it seems that
        the osa power must be above a certain threshold which seems to be about
        -50 dBm.
        """
        # Hacky way because we have not implemented a way to change the time
        # the OSA spends on a 0 nm span sweep
        osa.sweep()
        time.sleep(1)
        osa.stop_sweep()
        osa.update_spectrum()
        max_power = np.max(osa.powers)
        if max_power < min_idler_power:
            return False
        else:
            return True

    tisa.set_wavelength_iterative_method(tisa_wl, osa, error_tolerance=0.05)
    idler_loc = calculate_approx_idler_loc(tisa_wl, pump_wls, idler_side)
    osa_params_before_opt = {
        "res": osa.resolution,
        "sen": osa.sensitivity,
        "sweeptype": osa.sweeptype,
    }
    osa.resolution = 1
    osa.sensitivity = "SMID"
    osa.span = idler_loc
    osa.sweeptype = "RPT"
    reliable_idler_pow_for_pol_opt = "Idler power large enough for pol optimization"
    if check_idler_power(osa):
        print("Idler power too small for pol optimization")
        reliable_idler_pow_for_pol_opt = (
            "WARNING: Idler power too small for reliable pol optimization"
        )
    osa.sweep()
    optimize_multiple_pol_cons(
        arduino,
        *pol_cons,
        max_or_min=max_or_min,
        tolerance=tolerance,
    )
    osa.resolution = osa_params_before_opt["res"]
    osa.sensitivity = osa_params_before_opt["sen"]
    osa.sweeptype = osa_params_before_opt["sweeptype"]
    return reliable_idler_pow_for_pol_opt


# %% Sweep TISA measurements

phase_matching_approx_fit_loc = "./processing/fits/C_plus_L_band/linear_fit.txt"
phase_matching_approx_fit = np.poly1d(
    np.loadtxt(phase_matching_approx_fit_loc, delimiter=",")
)
#  Initial setup settings
ando1_start = 1577
ando2_start = 1563
pump_stepsize = 2.5
total_pump_move_nm = 30
num_pump_steps = int(total_pump_move_nm / pump_stepsize) + 1

pump_wl_list = [
    (ando1_start + pump_stepsize * i, ando2_start - pump_stepsize * i)
    for i in range(num_pump_steps)
]

# If 0, then only the tisa point from linear fit will be measured
tisa_dist_either_side_of_peak = 0.8
tisa_stepsize = 0.1
num_tisa_steps = int(tisa_dist_either_side_of_peak / tisa_stepsize) * 2
idler_side = "red"
duty_cycles = [0.1, 0.2, 0.25, 0.5, 1]
osa_params = {
    "res": 0.05,
    "sens": "SMID",
}
verdi_power = 6
verdi.power = verdi_power
num_sweep_reps = 5
params = {
    "ando1_start": ando1_start,
    "ando2_start": ando2_start,
    "pump_stepsize": pump_stepsize,
    "total_pump_move_nm": total_pump_move_nm,
    "num_pump_steps": num_pump_steps,
    "pump_wl_list": pump_wl_list,
    "tisa_dist_either_side_of_peak": tisa_dist_either_side_of_peak,
    "tisa_stepsize": tisa_stepsize,
    "num_tisa_steps": num_tisa_steps,
    "idler_side": idler_side,
    "duty_cycles": duty_cycles,
    "osa_params": osa_params,
    "verdi_power": verdi_power,
    "num_sweep_reps": num_sweep_reps,
    "phase_match_fit": phase_matching_approx_fit_loc,
    "pulse_freq": pulse_freq,
    "mean_p_wl": np.mean(pump_wl_list[0]),
}
data_folder = (
    f"./data/pol_opt_often/c_plus_l_band/tisa_sweep/mean_p_wl={params['mean_p_wl']}"
)

sweep_w_pol_opt_based_on_linear_fit(
    params,
    osa_params,
    ando1_laser,
    ando2_laser,
    verdi,
    tisa,
    osa,
    pico,
    pol_con1,
    pol_con2,
    arduino,
    data_folder,
    ipg_edfa,
)
