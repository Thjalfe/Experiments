import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
from clients.dc_power_supply_clients import E3631A
from clients.diode_controller_clients import EtekMLDC1032
from clients.hp81200_client import HP81200Client
from clients.multimeters import Agilent34401A
from clients.power_meter_clients import Agilent8163x
from clients.rf_clock_clients import HP8341
from external_stay.processing.helper_funcs import (
    Fiber,
)

from external_stay.run_exp_funcs.bias import (
    start_scrambler,
    stop_scrambler,
)
from external_stay.run_exp_funcs.helper_funcs import DiodeControllerRouter
from external_stay.run_exp_funcs.helper_funcs import (
    calc_pulse_len,
    calc_segment_len_m,
    optimize_brill_freq_hill_climb,
    optimize_brill_freq_golden,
    print_time_status,
    set_multiple_seqs,
)


@dataclass
class MovingCollisionInputParams:
    pulse_gen_freq: float
    brill_pump_seg_len: int
    pump_seg_len: int
    stepsize_seg: int
    start_idx: int
    end_idx: int
    brill_freq_init_step: float
    brill_freq_min_step: float
    wl_p: float
    wl_s: float
    yenista_filter_loss_p: float
    yenista_filter_loss_fwm: float
    seg_len_m: float = field(init=False)
    stepsize_m: float = field(init=False)
    meters_scanned: float = field(init=False)
    brill_pump_start_idxs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seg_len_m = calc_segment_len_m(self.pulse_gen_freq)
        self.brill_pump_len_m = calc_pulse_len(
            self.pulse_gen_freq, self.brill_pump_seg_len
        )
        self.pump_len_m = calc_pulse_len(self.pulse_gen_freq, self.pump_seg_len)
        self.stepsize_m = self.seg_len_m * self.stepsize_seg
        self.meters_scanned = np.abs(self.start_idx - self.end_idx) * self.stepsize_m
        self.brill_pump_start_idxs = np.arange(
            self.start_idx, self.end_idx + self.stepsize_seg * 0.1, self.stepsize_seg
        )


@dataclass
class InstrumentSetup:
    brill_pump_seg_lens: list[int]
    brill_pump_module_idxs: list[int]
    seg_dist_between_pump_modules: int
    full_sequence_len: int
    fwm_pm: Agilent8163x
    brill_wl_pm: Agilent8163x
    rf_clock: HP8341
    pulse_gen: HP81200Client
    diode_controller: EtekMLDC1032 | DiodeControllerRouter
    pm_avg_time: float

    def __post_init__(self):
        self.fwm_pm.average_time = self.pm_avg_time
        self.brill_wl_pm.average_time = self.pm_avg_time


@dataclass
class CollisionMeasurementData:
    coords: np.ndarray
    coords_idxs: np.ndarray
    brill_powers: np.ndarray
    ref_brill_powers: np.ndarray
    opt_brill_freqs: np.ndarray
    fwm_power: np.ndarray
    stepsize_m: float
    pm_avg_time: float
    fut_IL: float
    fut_Ltot: float
    wl_p: float
    wl_s: float
    yenista_filter_loss_p: float
    yenista_filter_loss_fwm: float
    brill_pump_pulse_len: float
    pump_len: float

    @classmethod
    def initialize(
        cls,
        sweep: MovingCollisionInputParams,
        instr: InstrumentSetup,
        fut: Fiber,
    ):
        n = len(sweep.brill_pump_start_idxs)
        return cls(
            coords=(sweep.brill_pump_start_idxs - sweep.brill_pump_start_idxs[-1])
            * sweep.stepsize_m,
            coords_idxs=sweep.brill_pump_start_idxs,
            brill_powers=np.zeros(n),
            ref_brill_powers=np.zeros(n),
            opt_brill_freqs=np.zeros(n),
            fwm_power=np.zeros(n),
            stepsize_m=sweep.stepsize_m,
            pm_avg_time=instr.pm_avg_time,
            fut_IL=fut.insertion_loss,
            fut_Ltot=fut.length,
            wl_p=sweep.wl_p,
            wl_s=sweep.wl_p,
            yenista_filter_loss_p=sweep.yenista_filter_loss_p,
            yenista_filter_loss_fwm=sweep.yenista_filter_loss_fwm,
            brill_pump_pulse_len=sweep.brill_pump_len_m,
            pump_len=sweep.pump_len_m,
        )


def full_filepath_collision_meas(
    fiber: Fiber,
    input_params: MovingCollisionInputParams,
    base_path: str = "/home/thjalfe/Documents/PhD/Projects/Experiments/external_stay/data/moving_collisions",
    extra_path: str = "",
):
    fiber_str = f"{fiber.full_fiber_name}"
    if len(extra_path) > 0:
        base_path = f"{base_path}/{extra_path}"
    dir = f"{base_path}/{fiber_str}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    base_filename = (
        f"brill-pump-dur={input_params.brill_pump_len_m:.1f}m_"
        f"m-scanned={input_params.meters_scanned:.1f}m_"
        f"wl-p={input_params.wl_p:.1f}nm_wl-s={input_params.wl_s:.1f}nm_"
        f"stepsize-m={input_params.stepsize_m:.1f}m.pkl"
    )
    filepath = os.path.join(dir, base_filename)

    if not os.path.exists(filepath):
        return filepath

    # Append _n if file exists
    n = 1
    while True:
        alt_filename = base_filename.replace(".pkl", f"_{n}.pkl")
        alt_filepath = os.path.join(dir, alt_filename)
        if not os.path.exists(alt_filepath):
            return alt_filepath
        n += 1


def run_collision_sweep(
    sweep_params: MovingCollisionInputParams,
    instr: InstrumentSetup,
    data: CollisionMeasurementData,
    filename: str,
    save_data: bool,
    num_segments_for_pulsegen: int = 5,
) -> CollisionMeasurementData:
    start_idxs = sweep_params.brill_pump_start_idxs
    cur_brill_freq = instr.rf_clock.frequency
    assert isinstance(cur_brill_freq, float)
    t0 = time.time()
    for idx, pump_pulse_start_idx in enumerate(start_idxs):
        seq1 = np.arange(
            pump_pulse_start_idx, pump_pulse_start_idx + instr.brill_pump_seg_lens[0]
        )
        seq2_start_idx = pump_pulse_start_idx + instr.seg_dist_between_pump_modules
        seq2 = np.arange(seq2_start_idx, seq2_start_idx + instr.brill_pump_seg_lens[1])
        set_multiple_seqs(
            instr.pulse_gen,
            instr.full_sequence_len,
            [seq1, seq2],
            enable_output_after_change=False,
            module_ids=instr.brill_pump_module_idxs,
            num_segments=num_segments_for_pulsegen,
        )
        time.sleep(0.2)
        # opt_brill_freq = optimize_brill_freq(
        #     multimeter,
        #     rf_clock,
        #     initial_step=sweep_params.brill_freq_init_step,
        #     min_step=sweep_params.brill_freq_min_step,
        #     delay=0.3,
        # )
        opt_brill_freq = optimize_brill_freq_golden(
            instr.brill_wl_pm,
            instr.rf_clock,
            cur_brill_freq - sweep_params.brill_freq_init_step,
            cur_brill_freq + sweep_params.brill_freq_init_step,
            sweep_params.brill_freq_min_step,
            delay=0.1,
        )
        cur_brill_freq = opt_brill_freq
        time.sleep(instr.pm_avg_time * 2)
        brill_power = instr.brill_wl_pm.power
        fwm_power = instr.fwm_pm.power
        data.brill_powers[idx] = brill_power
        data.opt_brill_freqs[idx] = opt_brill_freq
        data.fwm_power[idx] = fwm_power
        print_time_status(idx, t0, len(start_idxs))
        # save data every 10 measurement
        if idx % 10 == 0:
            if save_data:
                with open(f"{filename}", "wb") as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # obtain reference brill powers for each frequency used by looping through it with pump turned off
    print("Now obtaining reference pump powers")
    instr.diode_controller.disable_edfa("EDFA1")
    t0 = time.time()
    old_avg_time = instr.brill_wl_pm.average_time
    assert isinstance(old_avg_time, float)
    new_avg_time = 0.2
    instr.brill_wl_pm.average_time = new_avg_time
    for idx, rf_freq in enumerate(data.opt_brill_freqs):
        instr.rf_clock.frequency = rf_freq
        time.sleep(new_avg_time * 2)
        ref_brill_power = instr.brill_wl_pm.power
        data.ref_brill_powers[idx] = ref_brill_power
        print(f"Done with {idx + 1} out of {len(data.opt_brill_freqs)}")
    instr.brill_wl_pm.average_time = old_avg_time
    print(f"It took {(time.time() - t0):.1f}s")
    if save_data:
        with open(f"{filename}", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


@dataclass
class MovingCollisionInputParamsScope:
    pulse_gen_freq: float
    brill_pump_seg_len: int
    pump_seg_len: int
    stepsize_seg: int
    start_idx: int
    end_idx: int
    brill_freq_init_step: float
    brill_freq_min_step: float
    wl_p: float
    wl_s: float
    yenista_filter_loss_p: float
    pump_attenuation_before_PD: float
    seg_len_m: float = field(init=False)
    stepsize_m: float = field(init=False)
    meters_scanned: float = field(init=False)
    brill_pump_start_idxs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seg_len_m = calc_segment_len_m(self.pulse_gen_freq)
        self.brill_pump_len_m = calc_pulse_len(
            self.pulse_gen_freq, self.brill_pump_seg_len
        )
        self.pump_len_m = calc_pulse_len(self.pulse_gen_freq, self.pump_seg_len)
        self.stepsize_m = self.seg_len_m * self.stepsize_seg
        self.meters_scanned = np.abs(self.start_idx - self.end_idx) * self.stepsize_m
        self.brill_pump_start_idxs = np.arange(
            self.start_idx, self.end_idx + self.stepsize_seg * 0.1, self.stepsize_seg
        )


@dataclass
class CollisionMeasurementDataScope:
    coords: np.ndarray
    coords_idxs: np.ndarray
    brill_powers: np.ndarray
    ref_brill_powers: np.ndarray
    opt_brill_freqs: np.ndarray
    pump_waveforms: list
    fwm_waveforms: list
    fwm_power: np.ndarray
    stepsize_m: float
    fut_IL: float
    fut_Ltot: float
    wl_p: float
    wl_s: float
    yenista_filter_loss_p: float
    yenista_filter_loss_fwm: float
    brill_pump_pulse_len: float
    pump_len: float

    @classmethod
    def initialize(
        cls,
        sweep: MovingCollisionInputParams,
        fut: Fiber,
    ):
        n = len(sweep.brill_pump_start_idxs)
        return cls(
            coords=(sweep.brill_pump_start_idxs - sweep.brill_pump_start_idxs[-1])
            * sweep.stepsize_m,
            coords_idxs=sweep.brill_pump_start_idxs,
            brill_powers=np.zeros(n),
            ref_brill_powers=np.zeros(n),
            opt_brill_freqs=np.zeros(n),
            pump_waveforms=[],
            fwm_waveforms=[],
            fwm_power=np.zeros(n),
            stepsize_m=sweep.stepsize_m,
            fut_IL=fut.insertion_loss,
            fut_Ltot=fut.length,
            wl_p=sweep.wl_p,
            wl_s=sweep.wl_p,
            yenista_filter_loss_p=sweep.yenista_filter_loss_p,
            yenista_filter_loss_fwm=sweep.yenista_filter_loss_fwm,
            brill_pump_pulse_len=sweep.brill_pump_len_m,
            pump_len=sweep.pump_len_m,
        )


def full_filepath_collision_meas_scope(
    fiber: Fiber,
    input_params: MovingCollisionInputParamsScope,
    base_path: str = "/home/thjalfe/Documents/PhD/Projects/Experiments/external_stay/data/moving_collisions",
    extra_path: str = "",
):
    fiber_str = f"{fiber.full_fiber_name}"
    if len(extra_path) > 0:
        base_path = f"{base_path}/{extra_path}"
    dir = f"{base_path}/{fiber_str}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    base_filename = (
        f"brill-pump-dur={input_params.brill_pump_len_m:.1f}m_"
        f"m-scanned={input_params.meters_scanned:.1f}m_"
        f"wl-p={input_params.wl_p:.1f}nm_wl-s={input_params.wl_s:.1f}nm_"
        f"stepsize-m={input_params.stepsize_m:.1f}m.pkl"
    )
    filepath = os.path.join(dir, base_filename)

    if not os.path.exists(filepath):
        return filepath

    # Append _n if file exists
    n = 1
    while True:
        alt_filename = base_filename.replace(".pkl", f"_{n}.pkl")
        alt_filepath = os.path.join(dir, alt_filename)
        if not os.path.exists(alt_filepath):
            return alt_filepath
        n += 1
