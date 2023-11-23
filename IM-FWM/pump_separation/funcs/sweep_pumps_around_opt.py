import numpy as np
import time
import os
import pickle
from osa_control import OSA
from picoscope2000 import PicoScope2000a
from pol_cons import optimize_multiple_pol_cons, PolCon
from arduino_pm import ArduinoADC
from typing import Optional
from laser_control import Laser, TiSapphire
from verdi_laser import VerdiLaser
from ipg_edfa import IPGEDFA
import datetime
from logging_utils import setup_logging, logging_message


def calculate_approx_idler_loc(tisa_wl: float, pump_wls: np.ndarray, idler_side: str):
    if idler_side == "red":
        return 1 / (1 / pump_wls[0] + 1 / tisa_wl - 1 / pump_wls[1])
    elif idler_side == "blue":
        return 1 / (1 / pump_wls[1] + 1 / tisa_wl - 1 / pump_wls[0])
    else:
        raise ValueError("idler_side must be either 'red' or 'blue'")


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


def set_manual_pol_opt_and_back_to_normal_settings(
    osa: OSA,
    picoscope: PicoScope2000a,
    idler_wl: float,
    osa_params: dict,
    span_before: tuple,
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
    input("Press Enter when polarization is optimized")
    osa.stop_sweep()
    osa.resolution = osa_params["res"]
    osa.sensitivity = osa_params["sens"]
    osa.span = span_before
    osa.sweeptype = "SGL"
    osa.update_spectrum()
    idler_power = np.max(osa.powers)
    osa.set_power_marker(4, idler_power)


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
    pol_opt_method: str = "auto",
    ipg_edfa: Optional[IPGEDFA] = None,
):
    logger = setup_logging(
        data_folder, "log.log", [pump_laser1, pump_laser2], verdi, [ipg_edfa]
    )
    logging_message(logger, f"Starting sweep at {datetime.datetime.now()}")
    pump_wl_list = params["pump_wl_list"]
    for pump_wl_idx in range(num_pump_steps):
        logging_message(
            logger, f"Starting sweep for pump wls {pump_wl_list[pump_wl_idx]}"
        )
        data_pump_wl_dict = {pump_wl: {} for pump_wl in pump_wl_list}
        data_pump_wl_dict["params"] = params
        ando1_wl = pump_wl_list[pump_wl_idx][0]
        ando2_wl = pump_wl_list[pump_wl_idx][1]
        ando1_wls = np.linspace(
            ando1_wl - params["pump1_wl_delta"],
            ando1_wl + params["pump1_wl_delta"],
            params["num_pump1_steps"],
        )
        ando2_wls = np.linspace(
            ando2_wl - params["pump2_wl_delta"],
            ando2_wl + params["pump2_wl_delta"],
            params["num_pump2_steps"],
        )
        tisa_wl = params["phase_match_fit"](np.abs(ando1_wl - ando2_wl))
        idler_wl_approx = calculate_approx_idler_loc(
            tisa_wl, pump_wl_list[pump_wl_idx], "red"
        )
        logging_message(
            f"Setting tisa and idler wl to {tisa_wl} and {idler_wl_approx} nm"
        )
        tisa_span = (tisa_wl - 1, idler_wl_approx + 1)
        pump_laser1.wavelength = ando1_wl
        pump_laser2.wavelength = ando2_wl
        tisa.set_wavelength_iterative_method(tisa_wl, osa, error_tolerance=0.05)
        if pol_opt_method == "auto":
            logging_message(logger, "Starting pol opt...")
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
        elif pol_opt_method == "manual":
            set_manual_pol_opt_and_back_to_normal_settings(
                osa,
                pico,
                idler_wl_approx,
                osa_params,
                tisa_span,
                pol_opt_dc=params["pol_opt_dc"],
            )
        osa.sweep()
        tisa_pow = np.max(osa.powers)
        osa.set_power_marker(3, tisa_pow)
        osa.set_power_marker(4, idler_power)
        logging_message(
            logger,
            f"CE after pol opt for pump wls {pump_wl_list} is: {tisa_pow - idler_power}",
        )
        if idler_power < -55:
            logging_message(logger, "Idler power too low, a pump must be off", "error")
        tisa_span_len = tisa_span[1] - tisa_span[0]
        num_samples = int(tisa_span_len / osa.resolution) * params["sampling_ratio"]
        if num_samples % 2 == 0:
            num_samples += 1
        if params["sampling_ratio"] == 0:
            num_samples = len(osa.wavelengths)
        spectra = {
            dc: np.zeros(
                (
                    len(ando1_wls),
                    len(ando2_wls),
                    params["num_sweep_reps"],
                    2,
                    num_samples,
                )
            )
            for dc in params["duty_cycles"]
        }
        spectra_dimension_explanation = [
            "ando1_wl_idx",
            "ando2_wl_idx",
            "sweep_rep",
            "wl/pow",
            "wl_idx",
        ]
        osa.samples = num_samples
        logging_message(logger, f"Starting sweep for pump wls {pump_wl_list}...")
        for i in range(params["num_sweep_reps"]):
            if i > 0:
                pump_laser1.wavelength = ando1_wl
                if pol_opt_method == "auto":
                    set_auto_pol_opt_and_back_to_normal_settings(
                        osa,
                        pico,
                        idler_wl_approx,
                        osa_params,
                        tisa_span,
                        pol_con1,
                        pol_con2,
                        pol_opt_dc=params["pol_opt_dc"],
                    )
                elif pol_opt_method == "manual":
                    set_manual_pol_opt_and_back_to_normal_settings(
                        osa,
                        pico,
                        idler_wl_approx,
                        osa_params,
                        tisa_span,
                        pol_opt_dc=params["pol_opt_dc"],
                    )
                if params["sampling_ratio"] != 0:
                    osa.samples = num_samples
            for ando1_wl_idx, ando1_wl_tmp in enumerate(ando1_wls):
                for ando2_wl_idx, ando2_wl_tmp in enumerate(ando2_wls):
                    pump_laser1.wavelength = ando1_wl_tmp
                    pump_laser2.wavelength = ando2_wl_tmp
                    for dc in params["duty_cycles"]:
                        pico.awg.set_square_wave_duty_cycle(params["pulse_freq"], dc)
                        time.sleep(2)
                        osa.sweep()
                        spectra[dc][
                            ando1_wl_idx, ando2_wl_idx, i, 0, :
                        ] = osa.wavelengths
                        spectra[dc][ando1_wl_idx, ando2_wl_idx, i, 1, :] = osa.powers
            logging_message(
                logger,
                f"Finished sweep for pump wls {pump_wl_list[pump_wl_idx]} for sweep rep {i} out of {params['num_sweep_reps']}",
            )
        data_to_save = {
            "spectra": spectra,
            "spectra_dimension_explanation": spectra_dimension_explanation,
            "params": params,
            "osa_params": osa_params,
            "ando1_wls": ando1_wls,
            "ando2_wls": ando2_wls,
        }
        filename = (
            f"pump_wl={pump_wl_list[pump_wl_idx]}_lband_pump_sweep_around_opt.pkl"
        )
        with open(os.path.join(data_folder, filename), "wb") as f:
            pickle.dump(data_to_save, f)
