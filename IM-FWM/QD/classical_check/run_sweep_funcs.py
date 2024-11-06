import numpy as np
from typing import cast
from dataclasses import dataclass, field, asdict
import time
import os
import pickle
import matplotlib.pyplot as plt
import shutil
import sys

# sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
# sys.path.append("/home/thjalfe/Documents/PhD/Projects/Experiments/InstrumentControl/InstrumentControl/LabInstruments")
from pol_cons import optimize_multiple_pol_cons, PolCon
from laser_control import Laser, TiSapphire
from verdi_laser import VerdiLaser
from osa_control import OSA
from picoscope2000 import PicoAWG
from arduino_pm import ArduinoADC


def increment_pump_wl(pump_laser: Laser, increment: float):
    cur_wl = pump_laser.wavelength
    pump_laser.wavelength = cur_wl + increment
    print(f"{pump_laser.wavelength:.3f}")


def calculate_approx_idler_loc(tisa_wl: float, pump_wls: np.ndarray):
    return 1 / (1 / pump_wls[0] + 1 / tisa_wl - 1 / pump_wls[1])


def find_sig_idler_powers(sig_wl, idler_wl, spectrum, tolerance_nm):
    tolerance_idx = int(tolerance_nm / (spectrum[0, 1] - spectrum[0, 0]))
    sig_wl_idx_guess = np.argmin(np.abs(spectrum[0] - sig_wl))
    idler_wl_idx_guess = np.argmin(np.abs(spectrum[0] - idler_wl))
    sig_wl_idx = (
        np.argmax(
            spectrum[
                1,
                sig_wl_idx_guess - tolerance_idx : sig_wl_idx_guess + tolerance_idx,
            ]
        )
        + sig_wl_idx_guess
        - tolerance_idx
    )
    idler_wl_idx = (
        np.argmax(
            spectrum[
                1,
                idler_wl_idx_guess - tolerance_idx : idler_wl_idx_guess + tolerance_idx,
            ]
        )
        + idler_wl_idx_guess
        - tolerance_idx
    )

    sig_power = spectrum[1, sig_wl_idx]
    idler_power = spectrum[1, idler_wl_idx]
    return sig_power, idler_power


def run_pol_opt(
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
    old_sweep_type = osa.sweeptype
    old_res = osa.resolution
    old_level_scale = osa.level_scale
    old_level = osa.level
    osa.sweeptype = "RPT"
    osa.resolution = 0.5
    osa.span = idler_wl
    osa.level_scale = level_scale
    osa.level = (
        idler_power
        - (level_scale * 2)
        + (8 - num_divisions_below_start_power) * level_scale
    )
    # osa sets level such that there are two divisions above the set level
    # there are 8 divisions in total
    osa.sweep()
    if pol_opt_method == "manual":
        input("Press enter when polarization is optimized")
    elif pol_opt_method == "auto":
        assert (
            pm is not None
        ), "pm should be an instance of the ArduinoADC class if pol_opt_method is 'auto'"
        assert (
            pol_con1 is not None
        ), "pol_con1 should be an instance of the PolCon class if pol_opt_method is 'auto'"
        assert (
            pol_con2 is not None
        ), "pol_con2 should be an instance of the PolCon class if pol_opt_method is 'auto'"
        optimize_multiple_pol_cons(pm, pol_con1, pol_con2, tolerance=0.5)
    osa.resolution = old_res
    osa.level_scale = old_level_scale
    osa.level = old_level
    osa.sweeptype = old_sweep_type


def optimize_polarization_return_ce(
    osa,
    pump2,
    p1_wl,
    p2_max_ce,
    sig_wl,
    dc_array,
    tolerance_nm,
    pico_awg,
    pulse_freq,
    level_scale=2,
    num_divisions_below_start_power=2,
    num_reps=5,
    idler_side="red",
    pol_opt_method="manual",
    pm=None,
    pol_con1=None,
    pol_con2=None,
):
    old_res = osa.resolution
    osa.resolution = old_res / 2
    osa.stop_sweep()
    osa.sweeptype = "SGL"
    osa.sweep()
    num_wls = len(osa.wavelengths)
    spectra_pol_opt = np.zeros((len(dc_array), num_reps, 2, num_wls))
    ce_peak_pol_opt = np.zeros((len(dc_array), num_reps))
    for dc_idx, dc in enumerate(dc_array):
        pico_awg.set_square_wave_duty_cycle(pulse_freq, dc)
        time.sleep(1)

        pump2.wavelength = p2_max_ce
        idler_loc = calculate_approx_idler_loc(
            sig_wl,
            np.array([p1_wl, p2_max_ce]),
        )
        if idler_side == "red":
            osa.span = (sig_wl - 1, idler_loc + 1)
        elif idler_side == "blue":
            osa.span = (idler_loc - 1, sig_wl + 1)
        osa.set_wavelength_marker(1, sig_wl)
        osa.set_wavelength_marker(2, idler_loc)
        osa.sweep()
        sig_power, idler_power = find_sig_idler_powers(
            sig_wl,
            idler_loc,
            np.vstack((osa.wavelengths, osa.powers)),
            tolerance_nm,
        )
        osa.set_power_marker(3, sig_power)
        osa.set_power_marker(4, idler_power)
        print(
            f"Optimizing polarization for pump1_wl: {p1_wl:.3f}, CE peak before pol opt: {idler_power - sig_power - 10*np.log10(dc):.3f} with DC: {dc*100:.2f}%"
        )
        run_pol_opt(
            osa,
            idler_loc,
            idler_power,
            level_scale=level_scale,
            num_divisions_below_start_power=num_divisions_below_start_power,
            pol_opt_method=pol_opt_method,
            pm=pm,
            pol_con1=pol_con1,
            pol_con2=pol_con2,
        )

        if idler_side == "red":
            osa.span = (sig_wl - 1, idler_loc + 1)
        elif idler_side == "blue":
            osa.span = (idler_loc - 1, sig_wl + 1)
        if dc_idx == 0:
            old_res = osa.resolution
        for i in range(num_reps):
            spectrum_pol_opt = np.vstack((osa.wavelengths, osa.powers))
            spectra_pol_opt[dc_idx, i, :, :] = spectrum_pol_opt
            sig_power, idler_power = find_sig_idler_powers(
                sig_wl,
                idler_loc,
                spectrum_pol_opt,
                tolerance_nm,
            )
            ce_peak_pol_opt[dc_idx, i] = idler_power - sig_power - 10 * np.log10(dc)
            if i == num_reps - 1:
                break
            osa.sweep()
    for dc_idx, dc in enumerate(dc_array):
        print(f"DC: {dc*100:.1f}%")
        print(
            f"CE peak (mean of {num_reps} measurements) after pol opt: {np.mean(ce_peak_pol_opt[dc_idx]):.3f}"
        )
    osa.resolution = old_res
    return ce_peak_pol_opt, spectra_pol_opt


def optimize_one_pump_wl(
    pump2_laser: Laser,
    osa: OSA,
    pump1_wl: float,
    sig_wl: float,
    pump2_wl_start: float,
    pump2_wl_stop: float,
    stepsize: float,
    osa_res: float = 0.1,
    idler_tolerance_nm: float = 0.25,
    plot_ce_progression: bool = False,
    save_data: bool = False,
    data_loc: str | None = None,
    dc: float = 0.05,
    set_pump_to_max_ce: bool = True,
    idler_side: str = "red",
) -> tuple[list, list, float]:
    def post_processing(
        osa,
        ce,
        pump2_array,
        pump1_wl,
        sig_wl,
        spectra,
        idler_tolerance_nm,
        set_pump_to_max_ce,
        idler_side,
    ):
        max_ce = np.max(ce)
        pump2_max_ce = pump2_array[np.argmax(ce)]
        idler_loc_max = calculate_approx_idler_loc(
            sig_wl, np.array([pump1_wl, pump2_max_ce])
        )
        sig_power, idler_power = find_sig_idler_powers(
            sig_wl, idler_loc_max, spectra[np.argmax(ce)], idler_tolerance_nm
        )
        print(f"Max CE: {max_ce:.3f} at pump2_wl: {pump2_max_ce:.3f}")
        print(f"Idler loc at max CE: {idler_loc_max:.3f}")
        print(
            f"This results in a separation between signal and idler of {idler_loc_max - sig_wl:.3f} nm"
        )
        if set_pump_to_max_ce:
            print(f"Setting pump 2 wl to {pump2_max_ce:.3f}")
            pump2_laser.wavelength = pump2_max_ce
            if idler_side == "red":
                osa.span = (sig_wl - 1, idler_loc_max + 1)
            elif idler_side == "blue":
                osa.span = (idler_loc_max - 1, sig_wl + 1)
            time.sleep(1)
            osa.set_wavelength_marker(1, sig_wl)
            osa.set_wavelength_marker(2, idler_loc_max)
            osa.set_power_marker(3, sig_power)
            osa.set_power_marker(4, idler_power)
        return max_ce, pump2_max_ce, idler_loc_max

    osa.level_scale = 10
    osa.resolution = osa_res
    ce = []
    ce_offset = -10 * np.log10(dc)
    osa.stop_sweep()
    osa.sweeptype = "SGL"
    pump2_array = np.arange(pump2_wl_start, pump2_wl_stop + stepsize, stepsize)
    spectra = []
    try:
        for pump2_wl_iter, pump2_wl in enumerate(pump2_array):

            idler_loc = calculate_approx_idler_loc(
                sig_wl, np.array([pump1_wl, pump2_wl])
            )
            pump2_laser.wavelength = pump2_wl
            if idler_side == "red":
                osa.span = (sig_wl - 1, idler_loc + 1)
            elif idler_side == "blue":
                osa.span = (idler_loc - 1, sig_wl + 1)

            osa.sweep()
            spectrum = np.vstack((osa.wavelengths, osa.powers))
            sig_power, idler_power = find_sig_idler_powers(
                sig_wl, idler_loc, spectrum, idler_tolerance_nm
            )
            ce_tmp = idler_power - sig_power + ce_offset
            print(f"CE: {ce_tmp:.3f} for pump2_wl: {pump2_wl:.3f}")
            spectra.append(spectrum)
            ce.append(float(ce_tmp))
            if plot_ce_progression and pump2_wl_iter > 0:
                plt.plot(pump2_array[: pump2_wl_iter + 1], ce)
                plt.xlabel("Pump2 wavelength (nm)")
                plt.ylabel(r"CE$_{peak}$ (dB)")
                plt.title(
                    rf"Max CE$_{{peak}}$ so far: {np.max(ce):.2f} at pump2_wl: {pump2_array[np.argmax(ce)]:.2f}"
                )
                plt.show()
    except KeyboardInterrupt:
        max_ce, pump2_max_ce, idler_loc_max = post_processing(
            osa,
            ce,
            pump2_array,
            pump1_wl,
            sig_wl,
            spectra,
            idler_tolerance_nm,
            set_pump_to_max_ce,
            idler_side,
        )
    finally:
        max_ce, pump2_max_ce, idler_loc_max = post_processing(
            osa,
            ce,
            pump2_array,
            pump1_wl,
            sig_wl,
            spectra,
            idler_tolerance_nm,
            set_pump_to_max_ce,
            idler_side,
        )
        if save_data:
            extra_dir_name = f"ce_sweep_const_p1_wl={pump1_wl:.2f}_sig_wl={sig_wl:.2f}_p2_wl={pump2_wl_start:.2f}-{pump2_wl_stop:.2f}_stepsize={stepsize:.2f}"
            assert data_loc is not None, "data_loc should be a valid directory path"
            if not os.path.exists(os.path.join(data_loc, extra_dir_name)):
                os.makedirs(os.path.join(data_loc, extra_dir_name))
            data_dict = {
                "spectra": spectra,
                "ce": ce,
                "pump2_wls": pump2_array,
                "max_ce": max_ce,
                "pump2_max_ce": pump2_max_ce,
                "idler_loc_max": idler_loc_max,
                "dc": dc,
            }
            fig, ax = plt.subplots()
            ax = cast(plt.Axes, ax)
            fig = cast(plt.Figure, fig)
            ax.plot(pump2_array, ce)
            ax.set_xlabel("Pump2 wavelength (nm)")
            ax.set_ylabel(r"CE$_{peak}$ (dB)")
            ax.set_title(
                rf"Max CE$_{{peak}}$: {max_ce:.2f} at $\lambda_{{p2}}$: {pump2_max_ce:.2f}, $\lambda_{{i}}-\lambda_{{sig}}$: {idler_loc_max - sig_wl:.1f} nm"
            )
            fig.savefig(
                os.path.join(data_loc, extra_dir_name, "ce_plot.pdf"),
                bbox_inches="tight",
            )
            with open(os.path.join(data_loc, extra_dir_name, "data.pkl"), "wb") as file:
                pickle.dump(data_dict, file)
        if save_data:
            if not data_loc:
                raise ValueError("Must specify data_loc if save_data is True")
            if not os.path.exists(data_loc):
                os.makedirs(data_loc)
    return ce, spectra, pump2_max_ce


def optimize_one_pump_wl_increasing_fineness(
    pump2_laser,
    osa,
    pump2_max_ce,
    pump1_wl,
    sig_wl,
    prev_stepsize,
    stepsizes,
    osa_res=0.1,
    tolerance_nm=0.5,
    plot_ce_progression=False,
    save_data=False,
    data_loc=None,
    dc=0.05,
    num_prev_res_to_cover=1,
    idler_side="red",
):
    p2_wl_max_ce_array = []
    ce_max_array = []
    spectra_finer_steps = []
    for stepsize in stepsizes:
        num_steps_around_max_ce = prev_stepsize / stepsize * num_prev_res_to_cover
        pump2_wl_start = pump2_max_ce - stepsize * num_steps_around_max_ce / 2
        pump2_wl_stop = pump2_max_ce + stepsize * num_steps_around_max_ce / 2
        ce_array, spectra, pump2_max_ce = optimize_one_pump_wl(
            pump2_laser,
            osa,
            pump1_wl,
            sig_wl,
            pump2_wl_start,
            pump2_wl_stop,
            stepsize,
            osa_res=osa_res,
            idler_tolerance_nm=tolerance_nm,
            plot_ce_progression=plot_ce_progression,
            save_data=save_data,
            data_loc=data_loc,
            dc=dc,
            idler_side=idler_side,
        )
        p2_wl_max_ce_array.append(pump2_max_ce)
        ce_max_array.append(np.max(ce_array))
        spectra_finer_steps.append(spectra)

        prev_stepsize = stepsize
    ce_max_finer_steps = np.max(ce_max_array)
    best_p2_wl = p2_wl_max_ce_array[np.argmax(ce_max_array)]
    return best_p2_wl, ce_max_finer_steps, spectra_finer_steps


@dataclass
class Instruments:
    pump1: Laser
    pump2: Laser
    sig_laser: Laser | TiSapphire
    osa: OSA
    awg: PicoAWG
    verdi: VerdiLaser | None = None
    adc: ArduinoADC | None = None
    pol_cons: list[PolCon] = field(default_factory=list)


@dataclass
class SweepParams:
    pump1_array: np.ndarray
    pump2_wl_min: float
    pump2_wl_max: float
    pump2_wl_stop_init: float
    pump1_stepsize: float
    pump2_stepsize: float
    min_distance_between_pumps_nm: float
    pump2_wl_move_factor: float
    increased_fineness_stepsizes: np.ndarray | None = None
    num_prev_res_to_cover: int = 1
    dc_array: np.ndarray = np.array([0.05])
    pump2_wl_start_init: float | None = None


@dataclass
class CommonSaveData:
    """Save data that is common for all pump 1 wavelengths"""

    duty_cycles: np.ndarray
    spectra_dims_after_pol_opt: str
    ce_dims_after_pol_opt: str


@dataclass
class CoarseSweepData:
    pump2_wls: np.ndarray
    ce_peak_vs_pump2_wl: list[np.ndarray]
    ce_avg_vs_pump2_wl: list[np.ndarray]
    spectra_vs_pump2_wl: list[np.ndarray]
    p2_max_ce: float


@dataclass
class PolOptData:
    """Data for each pump 1 wavelength"""

    p2_max_ce: float
    ce_peak_pol_opt: np.ndarray
    ce_avg_pol_opt: np.ndarray
    spectra_pol_opt: np.ndarray


@dataclass
class FinerSweepData:
    """Data for each pump 1 wavelength"""

    p2_max_ce: float
    ce_peak: np.ndarray
    ce_avg: np.ndarray
    spectra: list[np.ndarray]


@dataclass
class Pump1Data:
    """Data for each pump 1 wavelength"""

    coarse_sweep_data: CoarseSweepData
    pol_opt_data: PolOptData | None
    fine_sweep_data: FinerSweepData | None = None


@dataclass
class SaveData:
    common_data: CommonSaveData
    pump1_sweep_data: dict[float, Pump1Data]


def save_at_end_of_p1_iter(
    data: SaveData,
    sweep_params: SweepParams,
    data_loc: str,
    p1_wl_cur: float,
    old_dir_name: str | None = None,
):
    save_data_dict = asdict(data)
    pump1_wl_start = sweep_params.pump1_array[0]
    coarse_data = data.pump1_sweep_data[p1_wl_cur].coarse_sweep_data
    pump2_wl_start = coarse_data.pump2_wls[0]
    pump2_wl_stop = coarse_data.pump2_wls[1]
    if old_dir_name is not None:
        shutil.rmtree(os.path.join(data_loc, old_dir_name))
    extra_dir_name = f"p1_wl={pump1_wl_start:.1f}-{p1_wl_cur:.1f}_p2_wl={pump2_wl_start:.1f}-{pump2_wl_stop:.1f}_p1_stepsize={sweep_params.pump1_stepsize:.2f}_p2_stepsize={sweep_params.pump2_stepsize:.2f}"
    old_dir_name = extra_dir_name
    if not os.path.exists(os.path.join(data_loc, extra_dir_name)):
        os.makedirs(os.path.join(data_loc, extra_dir_name))
    with open(os.path.join(data_loc, extra_dir_name, "data.pkl"), "wb") as file:
        pickle.dump(save_data_dict, file)
    for p1_wl_tmp in sweep_params.pump1_array:
        if p1_wl_tmp <= p1_wl_cur:
            coarse_tmp = data.pump1_sweep_data[p1_wl_tmp].coarse_sweep_data
            fig, ax = plt.subplots()
            ax = cast(plt.Axes, ax)
            fig = cast(plt.Figure, fig)
            ax.plot(
                coarse_tmp.pump2_wls,
                coarse_tmp.ce_peak_vs_pump2_wl,
            )
            ax.set_xlabel("Pump2 wavelength (nm)")
            ax.set_ylabel(r"CE$_{peak}$ (dB)")
            ax.set_title(
                rf"Max CE$_{{peak}}$: {np.max(coarse_tmp.ce_peak_vs_pump2_wl):.2f} at pump1_wl: {p1_wl_tmp:.2f} and pump2_wl: {coarse_tmp.p2_max_ce:.2f}"
            )
            fig.savefig(
                os.path.join(data_loc, extra_dir_name, f"ce_plot_{p1_wl_tmp:.2f}.pdf"),
                bbox_inches="tight",
            )


def print_progress_at_end_of_p1_iter(data: SaveData):
    def pump_wls_to_thz_sep(p1wl, p2wl, c=299792458):
        p1wl_thz = c / (p1wl * 10**-9) * 10**-12
        p2wl_thz = c / (p2wl * 10**-9) * 10**-12
        return np.abs(p1wl_thz - p2wl_thz)

    pump_sep_thz = []
    ce_pol_opt_peak_tmp = []
    for p1_tmp in data.pump1_sweep_data.keys():
        if type(p1_tmp) is not np.float64:
            continue
        pol_opt_data = data.pump1_sweep_data[p1_tmp].pol_opt_data
        if pol_opt_data is None:
            continue
        print(p1_tmp)
        pump_sep_thz.append(
            pump_wls_to_thz_sep(
                p1_tmp,
                pol_opt_data.p2_max_ce,
            ),
        )
        print(np.mean(pol_opt_data.ce_peak_pol_opt[0]))
        ce_pol_opt_peak_tmp.append(np.mean(pol_opt_data.ce_peak_pol_opt[0]))
    print(f"Separation so far: {np.round(pump_sep_thz,3)} THz")
    print(f"CE peak so far: {np.round(ce_pol_opt_peak_tmp, 3)} dB")


def sweep_both_pumps(
    instruments: Instruments,
    sweep_params: SweepParams,
    sig_wl,
    osa_res=0.1,
    tolerance_nm=0.25,
    plot_ce_progression=True,
    save_data=True,
    data_loc=None,
    plot_sub_ce_progression=False,
    save_sub_data=False,
    data_sub_loc=None,
    turn_off_after_sweep=False,
    pulse_freq=10**5,
    pol_opt_method="manual",
) -> dict:
    if save_data:
        if not data_loc:
            raise ValueError("Must specify data_loc if save_data is True")
        if not os.path.exists(data_loc):
            os.makedirs(data_loc)
    data = SaveData(
        common_data=CommonSaveData(
            duty_cycles=sweep_params.dc_array,
            spectra_dims_after_pol_opt="dc x num_reps x 2 x num_wls",
            ce_dims_after_pol_opt="dc x num_reps",
        ),
        pump1_sweep_data={},
    )
    pol_opt_data = None
    fine_sweep_data = None
    pump2_max_ce = None
    fine_step_p2_max_ce = None
    old_dir_name = None
    pump2_wl_stop = None
    dc_tmp_for_quick_opt = sweep_params.dc_array[0]
    pump2_wl_start = sweep_params.pump2_wl_start_init

    try:
        for p1_wl_idx, p1_wl in enumerate(sweep_params.pump1_array):
            if p1_wl > sweep_params.pump2_wl_start_init:
                idler_side = "red"
            else:
                idler_side = "blue"
            instruments.awg.set_square_wave_duty_cycle(pulse_freq, dc_tmp_for_quick_opt)
            pump2_wl_stop_local = (
                sweep_params.pump2_wl_stop_init
            )  # Does nothing but fixes linting error
            if p1_wl_idx == 0:
                pump2_wl_stop = sweep_params.pump2_wl_stop_init
                pump2_wl_stop_local = pump2_wl_stop
            if idler_side == "red":
                if pump2_wl_stop > p1_wl - sweep_params.min_distance_between_pumps_nm:
                    pump2_wl_stop_local = (
                        p1_wl - sweep_params.min_distance_between_pumps_nm
                    )
                else:
                    pump2_wl_stop_local = sweep_params.pump2_wl_max
            elif idler_side == "blue":
                if pump2_wl_start < p1_wl + sweep_params.min_distance_between_pumps_nm:
                    pump2_wl_start = p1_wl + sweep_params.min_distance_between_pumps_nm

            if fine_step_p2_max_ce is not None:
                pump2_wl_stop_local = fine_step_p2_max_ce
            elif pump2_max_ce is not None:
                pump2_wl_stop_local = pump2_max_ce
            if isinstance(instruments.sig_laser, Laser):
                instruments.sig_laser.wavelength = sig_wl
                instruments.sig_laser.adjust_wavelength(instruments.osa)

            assert pump2_wl_start is not None, "pump2_wl_start should not be None here"
            pump2_array = np.arange(
                pump2_wl_start,
                pump2_wl_stop_local + sweep_params.pump2_stepsize,
                sweep_params.pump2_stepsize,
            )
            instruments.pump1.wavelength = p1_wl
            ce, spectra, p2_max_ce = optimize_one_pump_wl(
                instruments.pump2,
                instruments.osa,
                p1_wl,
                sig_wl,
                pump2_wl_start,
                pump2_wl_stop_local,
                sweep_params.pump2_stepsize,
                osa_res=osa_res,
                idler_tolerance_nm=tolerance_nm,
                plot_ce_progression=plot_sub_ce_progression,
                save_data=save_sub_data,
                data_loc=data_sub_loc,
                dc=dc_tmp_for_quick_opt,
                set_pump_to_max_ce=True,
                idler_side=idler_side,
            )
            ce_avg = ce + 10 * np.log10(dc_tmp_for_quick_opt)
            coarse_sweep_data = CoarseSweepData(
                pump2_wls=pump2_array,
                ce_peak_vs_pump2_wl=ce,
                ce_avg_vs_pump2_wl=ce_avg,
                spectra_vs_pump2_wl=spectra,
                p2_max_ce=p2_max_ce,
            )

            # when prev loop is done, start p2 sweep from below the max ce from prev step
            pump2_wl_start = (
                p2_max_ce
                - sweep_params.pump1_stepsize * sweep_params.pump2_wl_move_factor
            )

            if sweep_params.increased_fineness_stepsizes is not None:
                fine_step_p2_max_ce, fine_step_max_ce, fine_step_spectra = (
                    optimize_one_pump_wl_increasing_fineness(
                        instruments.pump2,
                        instruments.osa,
                        p2_max_ce,
                        p1_wl,
                        sig_wl,
                        sweep_params.pump2_stepsize,
                        sweep_params.increased_fineness_stepsizes,
                        osa_res=0.1,
                        tolerance_nm=0.5,
                        plot_ce_progression=True,
                        save_data=False,
                        data_loc=data_loc,
                        dc=dc_tmp_for_quick_opt,
                        num_prev_res_to_cover=sweep_params.num_prev_res_to_cover,
                        idler_side=idler_side,
                    )
                )
                p2_max_ce = fine_step_p2_max_ce
                fine_step_max_ce_peak = fine_step_max_ce + 10 * np.log10(
                    dc_tmp_for_quick_opt
                )
                fine_sweep_data = FinerSweepData(
                    p2_max_ce=fine_step_p2_max_ce,
                    ce_peak=fine_step_max_ce,
                    ce_avg=fine_step_max_ce_peak,
                    spectra=fine_step_spectra,
                )
            if (pol_opt_method == "manual") or (pol_opt_method == "auto"):

                ce_pol_opt_peak, spec_pol_opt = optimize_polarization_return_ce(
                    instruments.osa,
                    instruments.pump2,
                    p1_wl,
                    fine_step_p2_max_ce,
                    sig_wl,
                    sweep_params.dc_array,
                    tolerance_nm,
                    instruments.awg,
                    pulse_freq,
                    idler_side=idler_side,
                    pol_opt_method=pol_opt_method,
                    pm=instruments.adc,
                    pol_con1=instruments.pol_cons[0],
                    pol_con2=instruments.pol_cons[1],
                )
                ce_pol_opt_avg = np.zeros_like(ce_pol_opt_peak)
                for dc_idx, dc in enumerate(sweep_params.dc_array):
                    ce_pol_opt_avg[dc_idx] = ce_pol_opt_peak[dc_idx] + 10 * np.log10(dc)
                pol_opt_data = PolOptData(
                    ce_peak_pol_opt=ce_pol_opt_peak,
                    ce_avg_pol_opt=ce_pol_opt_avg,
                    spectra_pol_opt=spec_pol_opt,
                    p2_max_ce=p2_max_ce,
                )
            if plot_ce_progression:
                fig, ax = plt.subplots()
                ax = cast(plt.Axes, ax)
                ax.plot(pump2_array, ce)
                ax.set_xlabel("Pump2 wavelength (nm)")
                ax.set_ylabel(r"CE$_{peak}$ (dB)")
                ax.set_title(
                    rf"Max CE$_{{peak}}$: {np.max(ce):.2f} at pump1_wl: {p1_wl:.2f} and pump2_wl: {p2_max_ce:.2f}, after pol opt with DC: {sweep_params.dc_array[0]:.2f}"
                )
                plt.show()
            data.pump1_sweep_data[p1_wl] = Pump1Data(
                coarse_sweep_data=coarse_sweep_data,
                pol_opt_data=pol_opt_data,
                fine_sweep_data=fine_sweep_data,
            )

            if save_data:
                assert (
                    data_loc is not None
                ), "data_loc should be a valid directory path if save_data is True"
                save_at_end_of_p1_iter(
                    data, sweep_params, data_loc, p1_wl, old_dir_name
                )
            print_progress_at_end_of_p1_iter(data)

    finally:
        if turn_off_after_sweep:
            instruments.pump1.disable()
            instruments.pump2.disable()
            if instruments.verdi is not None:
                instruments.verdi.shutter = 0
    return asdict(data)
