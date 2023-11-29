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
from arduino_pm import ArduinoADC
from ipg_edfa import IPGEDFA
from logging_utils import setup_logging, logging_message
from funcs.utils import extract_sig_wl_and_ce_multiple_spectra


def calculate_approx_idler_loc(tisa_wl: float, pump_wls: np.ndarray, idler_side: str):
    if idler_side == "red":
        return 1 / (1 / pump_wls[0] + 1 / tisa_wl - 1 / pump_wls[1])
    elif idler_side == "blue":
        return 1 / (1 / pump_wls[1] + 1 / tisa_wl - 1 / pump_wls[0])
    else:
        raise ValueError("idler_side must be either 'red' or 'blue'")


def get_ce_and_locs_from_spectra(
    spectra: dict, pump_wl_pair: tuple, num_reps: int, duty_cycles: List[float]
):
    idler_wl_2d = [[] for _ in duty_cycles]
    sig_wl_2d = [[] for _ in duty_cycles]
    ce_2d = [[] for _ in duty_cycles]
    for dc_idx, dc in enumerate(duty_cycles):
        idler_wl_lst = []
        sig_wl_lst = []
        ce_lst = []
        spectra_for_dc = spectra[dc]
        for rep in range(num_reps):
            if len(np.shape(spectra_for_dc)) == 4:
                spectra_sgl_rep = np.transpose(spectra_for_dc[rep], (0, 2, 1))
            else:
                spectra_sgl_rep = np.transpose(spectra_for_dc, (0, 2, 1))
            sig_wl_tmp, ce_tmp, idler_wl_tmp = extract_sig_wl_and_ce_multiple_spectra(
                spectra_sgl_rep, list(pump_wl_pair), np.shape(spectra_sgl_rep)[0]
            )  # The dimension that is being returned over is the tisa sweep across opt
            ce_lst.append(-ce_tmp)
            sig_wl_lst.append(sig_wl_tmp)
            idler_wl_lst.append(idler_wl_tmp)
        ce_lst = np.mean(ce_lst, axis=0)
        sig_wl_lst = np.mean(sig_wl_lst, axis=0)
        idler_wl_lst = np.mean(idler_wl_lst, axis=0)
        idler_wl_2d[dc_idx] = idler_wl_lst
        sig_wl_2d[dc_idx] = sig_wl_lst
        ce_2d[dc_idx] = ce_lst
    ce_mean_all_dcs = np.mean(ce_2d, axis=0)
    sig_wl_mean_all_dcs = np.mean(sig_wl_2d, axis=0)
    idler_wl_mean_all_dcs = np.mean(idler_wl_2d, axis=0)
    ce_max = np.max(ce_mean_all_dcs)
    ce_max_idx = np.argmax(ce_mean_all_dcs)
    sig_wl_max = sig_wl_mean_all_dcs[ce_max_idx]
    idler_wl_max = idler_wl_mean_all_dcs[ce_max_idx]
    return ce_max, sig_wl_max, idler_wl_max


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


def sweep_tisa(
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
        idler_loc: float,
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

    init_tisa_sweep(
        start_tisa_wl,
        stepsize,
        idler_side,
        idler_loc,
        osa_params,
        tisa,
        osa,
    )
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


def sweep_tisa_multiple_pump_seps(
    params: dict,
    osa_params: dict,
    pump_laser1: Laser,
    pump_laser2: Laser,
    verdi: VerdiLaser,
    tisa: TiSapphire,
    osa: OSA,
    pico: PicoScope2000a,
    data_folder: str,
    ipg_edfa: Optional[IPGEDFA] = None,
):
    logger = setup_logging(
        data_folder, "log.log", [pump_laser1, pump_laser2, verdi, [ipg_edfa]]
    )
    logging_message(logger, f"Starting sweep at {datetime.datetime.now()}")
    tisa_start_wl = params["start_wl"]
    for pump_wl_idx in range(len(params["pump_wl_list"])):
        pump1_wl = params["pump_wl_list"][pump_wl_idx][0]
        pump2_wl = params["pump_wl_list"][pump_wl_idx][1]
        logging_message(
            logger, f"Starting sweep for pump wls {params['pump_wl_list'][pump_wl_idx]}"
        )
        pump_laser1.wavelength = pump1_wl
        pump_laser2.wavelength = pump2_wl
        idler_wl_approx = calculate_approx_idler_loc(
            tisa_start_wl, params["pump_wl_list"][pump_wl_idx], "red"
        )
        logging_message(
            f"Setting tisa and idler wl to {tisa_start_wl} and {idler_wl_approx} nm"
        )
        tisa_span = (tisa_start_wl - 1, idler_wl_approx + 1)
        # Error tolerance higher than normal, but for this function, we do not know the
        # correct wavelengths beforehand so it does not matter

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
        spectra = sweep_tisa(
            tisa_start_wl,
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
            params["pulse_freq"],
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
        filename = f"pump_wl={params['pump_wl_list'][pump_wl_idx]}.pkl"
        with open(os.path.join(data_folder, filename), "wb") as f:
            pickle.dump(data_to_save, f)
        logging_message(
            logger,
            f"Saved data for pump wls {params['pump_wl_list'][pump_wl_idx]}",
        )
        ce_max, sig_wl_max, _ = get_ce_and_locs_from_spectra(
            spectra, params["pump_wl_list"][pump_wl_idx], params["num_sweep_reps"]
        )
        logging_message(
            logger,
            f"CE max for pump wls {params['pump_wl_list'][pump_wl_idx]} is {ce_max}",
        )
        logging_message(
            logger,
            f"Sig wl for max CE for pump wls {params['pump_wl_list'][pump_wl_idx]} is {sig_wl_max}",
        )
        tisa_start_wl = sig_wl_max
