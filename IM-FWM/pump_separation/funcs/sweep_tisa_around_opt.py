import numpy as np
import logging
import datetime
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


def optimize_pump_pols_at_tisa_wl_and_return_idler_pow(
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
    def return_idler_power(osa: OSA):
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
        return max_power

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
    idler_power = return_idler_power(osa)
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
    return idler_power


def set_auto_pol_opt_and_back_to_normal_settings(
    osa: OSA,
    picoscope: PicoScope2000a,
    idler_wl: float,
    osa_params: dict,
    span_before: tuple,
    pol_con1: PolCon,
    pol_con2: PolCon,
    arduino: ArduinoADC,
    pulse_freq: float,
    pol_opt_dc: float = 0.1,
):
    picoscope.awg.set_square_wave_duty_cycle(pulse_freq, pol_opt_dc)
    osa.samples = 0  # set to auto
    osa.span = idler_wl
    osa.sweeptype = "RPT"
    osa.resolution = 0.5
    print(f"Setting span to {idler_wl}")
    osa.sweep()
    optimize_multiple_pol_cons(arduino, pol_con1, pol_con2, tolerance=0.5)
    osa.stop_sweep()
    osa.resolution = osa_params["res"]
    osa.sensitivity = osa_params["sens"]
    osa.span = span_before
    osa.sweeptype = "SGL"
    osa.update_spectrum()
    idler_power = np.max(osa.powers)
    return idler_power


def move_tisa_until_no_hysteresis(
    stepsize: float, tisa: TiSapphire, osa: OSA, logger: logging.Logger
):
    logging_message(logger, "Moving TISA until no hysteresis...")
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
            logging_message(logger, "No longer hysteresis")
            break


def sweep_tisa_w_dutycycle(
    start_tisa_wl: float,
    stepsize: float,
    num_steps: int,
    idler_side: str,
    idler_loc: str,
    duty_cycles: List[float],
    osa_params: dict,
    num_sweep_reps: int,
    tisa: TiSapphire,
    osa: OSA,
    pico: PicoScope2000a,
    pulse_freq: float,
    logger: logging.Logger,
):
    def init_tisa_sweep(
        start_wl: float,
        stepsize: float,
        idler_side: str,
        osa_params: dict,
        tisa: TiSapphire,
        osa: OSA,
    ):
        logging_message(logger, "Initializing TISA sweep...")
        # -1*stepsize due to the hysterisis of the tisa
        tisa.set_wavelength_iterative_method(
            start_wl - stepsize, osa, error_tolerance=0.05
        )
        osa.span = (start_wl - 1, start_wl + 1)
        osa.resolution = osa_params["res"]
        osa.sensitivity = osa_params["sens"]
        move_tisa_until_no_hysteresis(stepsize, tisa, osa)
        if idler_side == "red":
            osa_span = (start_wl - 1, idler_loc + 1)
            osa.span = osa_span
        elif idler_side == "blue":
            osa_span = (idler_loc - 1, start_wl + 1)
            osa.span = osa_span
        logging_message(logger, "TISA sweep initialized")

    def init_no_tisa_sweep(
        center_wl: float,
        idler_loc: float,
        idler_side: str,
        osa_params: dict,
        osa: OSA,
    ):
        logging_message(logger, "Initializing no TISA sweep...")
        osa.resolution = osa_params["res"]
        osa.sensitivity = osa_params["sens"]
        if idler_side == "red":
            osa_span = (center_wl - 1, idler_loc + 1)
            osa.span = osa_span
        elif idler_side == "blue":
            osa_span = (idler_loc - 1, center_wl + 1)
            osa.span = osa_span
        logging_message(logger, "No TISA sweep initialized")

    if num_steps > 0:
        # If 0, then only the tisa point from linear fit will be measured
        init_tisa_sweep(start_tisa_wl, stepsize, idler_side, osa_params, tisa, osa)
    else:
        # stupid solution to avoid having to rewrite the code,
        # just set the tisa to the linear fit point
        init_no_tisa_sweep(
            start_tisa_wl,
            idler_loc,
            idler_side,
            osa_params,
            osa,
        )
        num_steps = 1
    spectrum_dict = {dc: [] for dc in duty_cycles}
    for i in range(num_steps):
        logging_message(logger, f"Starting TiSa sweep {i+1}/{num_steps}...")
        for dc in duty_cycles:
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
        logging_message(logger, f"TiSa sweep {i+1}/{num_steps} done!")
    return spectrum_dict


def sweep_w_pol_opt_based_on_linear_fit(
    params: dict,
    osa_params: dict,
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
    logger = setup_logging(
        data_folder, "log.log", [pump_laser1, pump_laser2], verdi, [ipg_edfa]
    )
    logging_message(logger, f"Starting sweep at {datetime.datetime.now()}")
    data_pump_wl_dict = {pump_wl: {} for pump_wl in params["pump_wl_list"]}
    data_pump_wl_dict["params"] = params
    for pump_wl_idx in range(len(params["pump_wl_list"])):
        pump1_wl = params["pump_wl_list"][pump_wl_idx][0]
        pump2_wl = params["pump_wl_list"][pump_wl_idx][1]
        logging_message(
            logger, f"Starting sweep for pump wls {params['pump_wl_list'][pump_wl_idx]}"
        )
        pump_laser1.wavelength = pump1_wl
        pump_laser2.wavelength = pump2_wl
        pump_wl_diff = np.abs(pump_laser1.wavelength - pump_laser2.wavelength)
        center_wl = params["phase_match_fit"](pump_wl_diff)
        tisa_wl = params["phase_match_fit"](np.abs(pump1_wl - pump2_wl))
        idler_wl_approx = calculate_approx_idler_loc(
            tisa_wl, params["pump_wl_list"][pump_wl_idx], "red"
        )
        logging_message(
            f"Setting tisa and idler wl to {tisa_wl} and {idler_wl_approx} nm"
        )
        tisa_span = (tisa_wl - 1, idler_wl_approx + 1)
        tisa.set_wavelength_iterative_method(tisa_wl, osa, error_tolerance=0.05)
        idler_power = set_auto_pol_opt_and_back_to_normal_settings(
            osa,
            pico,
            idler_wl_approx,
            osa_params,
            tisa_span,
            pol_con1,
            pol_con2,
            arduino,
            pol_opt_dc=params["pol_opt_dc"],
        )
        logging_message(logger, "Pol opt done!")
        osa.sweep()
        tisa_pow = np.max(osa.powers)
        osa.set_power_marker(3, tisa_pow)
        osa.set_power_marker(4, idler_power)
        logging_message(
            logger,
            f"CE after pol opt for pump wls {params['pump_wl_list'][pump_wl_idx]} is: {tisa_pow - idler_power}",
        )
        if idler_power < -55:
            logging_message(logger, "Idler power too low, a pump must be off", "error")

        tisa_span_len = tisa_span[1] - tisa_span[0]
        num_samples = int(tisa_span_len / osa.resolution) * params["sampling_ratio"]
        if num_samples % 2 == 0:
            num_samples += 1
        if params["sampling_ratio"] == 0:
            num_samples = len(osa.wavelengths)
        osa.samples = num_samples

        logging_message(
            logger,
            f"Starting TiSa sweep for pump wls {params['pump_wl_list'][pump_wl_idx]}...",
        )
        spectra = sweep_tisa_w_dutycycle(
            center_wl - params["tisa_dist_either_side_of_peak"],
            params["tisa_stepsize"],
            params["num_tisa_steps"],
            params["idler_side"],
            idler_wl_approx,
            params["duty_cycles"],
            params["osa_params"],
            params["num_sweep_reps"],
            tisa,
            osa,
            pico,
            logger,
        )
        logging_message(
            logger,
            f"TiSa sweep for pump wls {params['pump_wl_list'][pump_wl_idx]} done!",
        )
        data_to_save = {
            "spectra": spectra,
            "params": params,
            "osa_params": osa_params,
        }

        filename = f"pump_wl={params['pump_wl_list'][pump_wl_idx]}_lband_pump_sweep_around_opt.pkl"
        with open(os.path.join(data_folder, filename), "wb") as f:
            pickle.dump(data_to_save, f)
