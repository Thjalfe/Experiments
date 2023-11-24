import numpy as np
import pickle
import os
from typing import List, Optional
import time
from osa_control import OSA
from laser_control import TiSapphire, Laser
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from pol_cons import optimize_multiple_pol_cons, PolCon
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA
from logging_utils import setup_logging, logging_message


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
    *pol_cons: PolCon,
    max_or_min: str = "max",
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
    pulse_freq: float,
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


def sweep_w_pol_opt_based_on_linear_fit(
    params: dict,
    osa_params: dict,
    num_pump_steps: int,
    pump_laser1: Laser,
    pump_laser2: Laser,
    verdi: VerdiLaser,
    tisa: TiSapphire,
    osa: OSA,
    pico: PicoScope2000a,
    pol_con1: Optional[PolCon],
    pol_con2: Optional[PolCon],
    arduino: Optional[ArduinoADC],
    data_folder: str,
    ipg_edfa: Optional[IPGEDFA] = None,
):
    data_pump_wl_dict = {pump_wl: {} for pump_wl in params["pump_wl_list"]}
    data_pump_wl_dict["params"] = params
    for pump_wl_idx in range(num_pump_steps):
        pump_laser1.wavelength = params["pump_wl_list"][pump_wl_idx][0]
        pump_laser2.wavelength = params["pump_wl_list"][pump_wl_idx][1]
        pump_wl_diff = np.abs(pump_laser1.wavelength - pump_laser2.wavelength)
        center_wl = params["phase_match_fit"](pump_wl_diff)
        # Setting to 0.1 duty cycle to make sure the idler power is large enough for optimization
        # maybe later try to to optimize pol for each duty cycle
        pico.awg.set_square_wave_duty_cycle(params["pulse_freq"], 0.1)
        _ = optimize_pump_pols_at_tisa_wl(
            center_wl,
            params["pump_wl_list"][pump_wl_idx],
            params["idler_side"],
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
            np.array(params["pump_wl_list"][pump_wl_idx]),
            params["duty_cycles"],
            params["osa_params"],
            params["num_sweep_reps"],
            tisa,
            osa,
            pico,
        )
        data_to_save = {
            "spectra": spectra,
            "params": params,
            "osa_params": osa_params,
        }

        filename = f"pump_wl={params['pump_wl_list'][pump_wl_idx]}_lband_pump_sweep_around_opt.pkl"
        with open(os.path.join(data_folder, filename), "wb") as f:
            pickle.dump(data_to_save, f)
