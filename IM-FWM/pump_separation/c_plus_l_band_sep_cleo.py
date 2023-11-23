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
}
data_folder = "./data/C_plus_L_band/cleo/manual"


# %%  Perform the sweep
def move_tisa_until_no_hysteresis(stepsize: float, tisa: TiSapphire, osa: OSA):
    print("Moving TISA until no hysteresis...")
    osa.sweeptype = "SGL"
    osa.sweep()
    wavelengths = osa.wavelengths
    powers = osa.powers
    tisa_loc_prev = wavelengths[np.argmax(powers)]
    while True:
        tisa.delta_wl_nm(stepsize)
        osa.sweep()
        wavelengths = osa.wavelengths
        powers = osa.powers
        tisa_loc = wavelengths[np.argmax(powers)]
        tisa_loc_diff = tisa_loc - tisa_loc_prev
        if tisa_loc_diff > stepsize / 2:
            print("No longer hysteresis...")
            break


def sweep_tisa_w_dutycycle(
    start_tisa_wl: float,
    stepsize: float,
    num_steps: int,
    idler_side: str,
    pump_wls: np.ndarray,
    duty_cycles: List[float],
    osa_params: dict,
    num_sweep_reps: int,
    tisa: TiSapphire,
    osa: OSA,
    pico: PicoScope2000a,
):
    def init_tisa_sweep(
        start_wl: float,
        pump_wls: np.ndarray,
        stepsize: float,
        idler_side: str,
        osa_params: dict,
        tisa: TiSapphire,
        osa: OSA,
    ):
        print("Initializing TISA sweep...")
        # -1*stepsize due to the hysterisis of the tisa
        tisa.set_wavelength_iterative_method(
            start_wl - stepsize, osa, error_tolerance=0.05
        )
        osa.span = (start_wl - 1, start_wl + 1)
        osa.resolution = osa_params["res"]
        osa.sensitivity = osa_params["sens"]
        move_tisa_until_no_hysteresis(stepsize, tisa, osa)
        idler_loc = calculate_approx_idler_loc(start_wl, pump_wls, idler_side)
        if idler_side == "red":
            osa_span = (start_wl - 1, idler_loc + 1)
            osa.span = osa_span
        elif idler_side == "blue":
            osa_span = (idler_loc - 1, start_wl + 1)
            osa.span = osa_span
        print("TISA sweep initialized")

    def init_no_tisa_sweep(
        center_wl: float,
        pump_wls: np.ndarray,
        idler_side: str,
        osa_params: dict,
        osa: OSA,
    ):
        print("Initializing no TISA sweep...")
        osa.resolution = osa_params["res"]
        osa.sensitivity = osa_params["sens"]
        idler_loc = calculate_approx_idler_loc(center_wl, pump_wls, idler_side)
        if idler_side == "red":
            osa_span = (center_wl - 1, idler_loc + 1)
            osa.span = osa_span
        elif idler_side == "blue":
            osa_span = (idler_loc - 1, center_wl + 1)
            osa.span = osa_span
        print("OSA initialized for no TISA sweep meas")

    if num_steps > 0:
        # If 0, then only the tisa point from linear fit will be measured
        init_tisa_sweep(
            start_tisa_wl, pump_wls, stepsize, idler_side, osa_params, tisa, osa
        )
    else:
        # stupid solution to avoid having to rewrite the code,
        # just set the tisa to the linear fit point
        init_no_tisa_sweep(start_tisa_wl, pump_wls, idler_side, osa_params, tisa, osa)
        num_steps = 1
    spectrum_dict = {dc: [] for dc in duty_cycles}
    for i in range(num_steps):
        for i_dc, dc in enumerate(duty_cycles):
            sweeps_for_dc = (
                []
            )  # This will store the sweeps for a given dc for all repetitions
            pico.awg.set_square_wave_duty_cycle(pulse_freq, dc)
            for _ in range(num_sweep_reps):
                osa.sweep()
                wavelengths = osa.wavelengths
                powers = osa.powers
                spectrum = np.array([wavelengths, powers])
                sweeps_for_dc.append(spectrum)
            spectrum_dict[dc].append(sweeps_for_dc)
        tisa.delta_wl_nm(stepsize)
        osa.span = (osa.span[0] + stepsize, osa.span[1] + stepsize)
        print(f"TiSa sweep {i+1}/{num_steps} complete")
    return spectrum_dict


data_pump_wl_dict = {pump_wl: {} for pump_wl in pump_wl_list}
data_pump_wl_dict["params"] = params
for pump_step_iter in range(num_pump_steps):
    t0 = time.time()
    ando1_laser.wavelength = pump_wl_list[pump_step_iter][0]
    ando2_laser.wavelength = pump_wl_list[pump_step_iter][1]
    pump_wl_diff = np.abs(ando1_laser.wavelength - ando2_laser.wavelength)
    center_wl = phase_matching_approx_fit(pump_wl_diff)
    # Setting to 0.1 duty cycle to make sure the idler power is large enough for optimization
    # maybe later try to to optimize pol for each duty cycle
    pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.1)
    reliable_idler_pow_for_pol_opt = optimize_pump_pols_at_tisa_wl(
        center_wl,
        pump_wl_list[pump_step_iter],
        idler_side,
        tisa,
        osa,
        arduino,
        pol_con1,
        pol_con2,
    )
    spectra = sweep_tisa_w_dutycycle(
        center_wl - params["tisa_dist_either_side_of_peak"],
        params["tisa_stepsize"],
        params["num_tisa_steps"],
        params["idler_side"],
        np.array(params["pump_wl_list"][pump_step_iter]),
        params["duty_cycles"],
        params["osa_params"],
        params["num_sweep_reps"],
        tisa,
        osa,
        pico,
    )
    data_pump_wl_dict[pump_wl_list[pump_step_iter]]["spectra"] = spectra
    # data_pump_wl_dict[pump_wl_list[pump_step_iter]][
    #     "reliable_idler_pow_for_pol_opt"
    # ] = reliable_idler_pow_for_pol_opt
    print(
        f"Done with pump step {pump_step_iter+1}/{num_pump_steps} in {time.time()-t0}"
    )


verdi.shutter = 0
ipg_edfa.status = 0
ando1_laser.laser_off()
ando2_laser.laser_off()
with open(f"{data_folder}/data_tisa_sweep_real_polopt.pkl", "wb") as f:
    pickle.dump(data_pump_wl_dict, f)

# %%
