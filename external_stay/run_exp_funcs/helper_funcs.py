import re
from datetime import datetime, timedelta
import time
import numpy as np
from clients.rf_clock_clients import HP8341
from clients.hp81200_client import HP81200Client
from clients.diode_controller_clients import EtekMLDC1032
from clients.oscilloscope_clients import HP83480
from clients.osa_clients import OSAClient
from clients.power_meter_clients import Agilent8163x
from clients.scope_Agilent3000Xseries_client import Agilent3000X


def extract_sequence_names(input_string: str) -> list[str]:
    pattern = r"'([^']+)',0,0"
    matches = re.findall(pattern, input_string)
    if len(matches) == 0:
        pattern = r'"([^"]+)",0,0'
        matches = re.findall(pattern, input_string)
    return matches


def set_new_bin_seq(
    hp81200_base: HP81200Client,
    module_id: int,
    ones_pos: np.ndarray | list,
    seq_len: int,
    enable_output_after_change: bool = True,
    num_segments: int = 4,
):
    if isinstance(ones_pos, np.ndarray):
        ones_pos = ones_pos.astype(int)
        ones_pos = ones_pos.tolist()
    assert isinstance(ones_pos, list)
    # Need to turn off output to change sequences
    new_sequence = np.zeros(seq_len)
    new_sequence[ones_pos] = 1
    new_sequence = new_sequence.tolist()
    if hp81200_base.output:
        hp81200_base.output = False
    cur_seq_names = extract_sequence_names(hp81200_base.get_sequences())
    name_of_seq_to_change = cur_seq_names[module_id - 1]
    tmp_seq_names = cur_seq_names.copy()
    tmp_seq_names[module_id - 1] = "dummy"
    # Also need to change sequences that are changed to dummy
    hp81200_base.set_sequences(
        tmp_seq_names, sequence_length=seq_len, num_generators=num_segments
    )
    hp81200_base.update_binary_segment(name_of_seq_to_change, ones_pos, seq_len)
    hp81200_base.set_sequences(
        cur_seq_names, sequence_length=seq_len, num_generators=num_segments
    )
    hp81200_base.set_current_segments(new_sequence, module_id)
    if enable_output_after_change:
        hp81200_base.output = True


def set_multiple_seqs(
    hp81200_base: HP81200Client,
    seq_len: int,
    new_sequences: list,
    enable_output_after_change: bool = False,
    module_ids: list = [],
    num_segments: int = 5,
):
    if len(module_ids) == 0:
        module_ids = np.arange(1, len(new_sequences) + 1).tolist()

    for idx, seq in enumerate(new_sequences):
        module_id = module_ids[idx]
        set_one_seq(
            hp81200_base,
            module_id,
            seq,
            seq_len,
            enable_output_after_change=enable_output_after_change,
            num_segments=num_segments,
        )
    hp81200_base.output = True


def set_one_seq(
    hp81200_base: HP81200Client,
    module_id: int,
    ones_pos: int | list[int] | np.ndarray,
    seq_len: int,
    enable_output_after_change: bool = True,
    num_segments: int = 4,
):
    if isinstance(ones_pos, int):
        ones_pos = [ones_pos]

    set_new_bin_seq(
        hp81200_base,
        module_id,
        ones_pos,
        seq_len,
        enable_output_after_change=enable_output_after_change,
        num_segments=num_segments,
    )


def turn_on_edfas(diode_controllers: list[EtekMLDC1032]):
    for dc in diode_controllers:
        for edfa in dc.edfas.keys():  # type: ignore
            dc.set_edfa_to_max(edfa)


def turn_off_edfas(diode_controllers: list[EtekMLDC1032]):
    for dc in diode_controllers:
        for edfa in dc.edfas.keys():  # type: ignore
            dc.disable_edfa(edfa)


def get_scope_data(scope: HP83480, channel: int) -> np.ndarray:
    scope.waveform.channel = channel
    time_ax, voltage = scope.waveform.read_waveform()
    voltage[voltage > 1e10] = np.nan
    trace = np.vstack((time_ax, voltage))
    return trace


def find_square_region(trace, threshold_ratio=0.9, inverted: bool = False):
    if inverted:
        threshold = threshold_ratio * np.nanmin(trace)
        above_threshold = trace <= threshold

    else:
        threshold = threshold_ratio * np.nanmax(trace)
        above_threshold = trace >= threshold

    # Allow a single outlier by filling gaps of size 1
    fixed = above_threshold.copy()
    for i in range(1, len(fixed) - 1):
        if not fixed[i] and fixed[i - 1] and fixed[i + 1]:
            fixed[i] = True

    # Find the start and end of consecutive segments
    diffs = np.diff(np.concatenate(([0], fixed.astype(int), [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) > 0:
        # Select the longest segment (or modify to pick the first one)
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        return starts[best_idx], ends[best_idx]
    else:
        raise ValueError("No square region found")


def set_vertical_bars_around_peak_scope(scope, trace_num: int, threshold_ratio=0.3):
    trace = get_scope_data(scope, trace_num)
    x1, x2 = find_square_region(trace[1], threshold_ratio=threshold_ratio)
    x1 = trace[0][x1]
    x2 = trace[0][x2]
    x1 = x1 + scope.timebase.position
    x2 = x2 + scope.timebase.position
    scope.write(f"MARKER:X1POSITION {x1}")
    scope.write(f"MARKER:X2POSITION {x2}")


def center_scope_around_position(
    scope: HP83480, center_position: float, window_size: float
):
    scope.timebase.position = scope.timebase.position + center_position
    scope.timebase.scale = window_size / 10  # 10 divisions per window size
    scope.timebase.position = scope.timebase.position - 5 * scope.timebase.scale


def center_scope_around_peak(
    scope: HP83480,
    channel: int,
    window_width_factor: int,
    peak_thresh_ratio=0.7,
    inverted: bool = False,
):
    trace = get_scope_data(scope, channel)
    peak_left, peak_right = find_square_region(
        trace[1], threshold_ratio=peak_thresh_ratio, inverted=inverted
    )
    left_time = trace[0][peak_left]
    right_time = trace[0][peak_right]
    center_time = (left_time + right_time) / 2
    window_width = (right_time - left_time) * window_width_factor
    center_scope_around_position(scope, center_time, window_width)


def center_scope_around_max_val(
    scope: HP83480,
    channel: int,
    window_width: float,
    inverted: bool = False,
):
    trace = get_scope_data(scope, channel)
    if not inverted:
        center_time = trace[0][np.argmax(trace[1])]
    else:
        center_time = trace[0][np.argmin(trace[1])]

    center_scope_around_position(scope, center_time, window_width)


def get_osa_spec(osa: OSAClient) -> np.ndarray:
    return np.vstack((osa.wavelengths, osa.powers))


def increment_rf_clock(rf_clock: HP8341, amount: float):
    cur_freq = rf_clock.frequency
    assert isinstance(cur_freq, float)
    rf_clock.frequency = cur_freq + amount


def get_n_scope_traces(scope: HP83480, num_reps: int, channel: int) -> np.ndarray:
    traces = []
    for _ in range(num_reps):
        traces.append(get_scope_data(scope, channel))
    return np.array(traces)


def get_ones_pos(hp81200_base: HP81200Client, module_id: int) -> np.ndarray:
    cur_seg = np.array(hp81200_base.get_current_segments()[module_id - 1])
    ones_pos = np.where(cur_seg == 1)[0]
    return ones_pos


def increment_segment(
    hp81200_base: HP81200Client,
    module_id: int,
    increment: int,
    enable_output_after_change: bool = True,
    num_segments: int = 5,
):
    cur_seg = np.array(hp81200_base.get_current_segments()[module_id - 1])
    ones_pos = np.where(cur_seg == 1)[0]
    ones_pos = ones_pos + increment
    set_one_seq(
        hp81200_base,
        module_id,
        ones_pos,
        len(cur_seg),
        enable_output_after_change=enable_output_after_change,
        num_segments=num_segments,
    )


def increment_multiple_segments(
    hp81200_base: HP81200Client,
    module_ids: list[int],
    increment: int,
    num_segments: int = 5,
):
    for module_id in module_ids:
        increment_segment(
            hp81200_base,
            module_id,
            increment,
            enable_output_after_change=False,
            num_segments=num_segments,
        )
    hp81200_base.output = True


def optimize_brill_freq_hill_climb(
    pm: Agilent8163x,
    rf_clock: HP8341,
    initial_step: float = 10e6,
    min_step: float = 1e6,
    delay: float = 0.1,
    print_powers=False,
) -> float:
    current_freq = rf_clock.frequency
    assert isinstance(current_freq, float)
    step = initial_step

    pm_del = pm.average_time
    assert isinstance(pm_del, float)
    # Initial scan: measure at f - step, f, f + step
    trial_freqs = [
        round((current_freq - np.abs(step)) / 1e5) * 1e5,
        round(current_freq / 1e5) * 1e5,
        round((current_freq + np.abs(step)) / 1e5) * 1e5,
    ]
    trial_powers = []

    for f in trial_freqs:
        rf_clock.frequency = f
        time.sleep(delay + pm_del)
        trial_powers.append(pm.power)

    best_index = int(np.argmax(trial_powers))
    current_freq = trial_freqs[best_index]
    current_power = trial_powers[best_index]

    # Determine direction: are we going up or down in freq?
    if best_index == 0:
        step = -abs(step)
    elif best_index == 2:
        step = abs(step)
    else:
        # we were already at the best spot — arbitrarily go positive
        step = abs(step)
    freqs = []
    pow = []

    while abs(step) > min_step:
        next_freq = round((current_freq + step) / 1e5) * 1e5
        rf_clock.frequency = next_freq
        time.sleep(delay + pm_del)
        test_power = pm.power
        assert isinstance(test_power, float)
        if print_powers:
            print(test_power)

        if test_power > current_power:
            current_freq = next_freq
            current_power = test_power
            step *= 1.1
        else:
            step *= -0.5

        freqs.append(next_freq)
        pow.append(current_power)

    max_freq = freqs[np.argmax(pow)]
    rf_clock.frequency = max_freq
    return max_freq


def optimize_brill_freq_golden(
    pm: Agilent8163x,
    rf_clock: HP8341,
    low: float,
    high: float,
    tol: float = 1e6,
    delay: float = 0.1,
    print_trace: bool = False,
) -> float:
    invphi = (np.sqrt(5) - 1) / 2  # 0.618…
    invphi2 = (3 - np.sqrt(5)) / 2  # 0.382…

    a, b = low, high
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)

    def measure(f):
        rf_clock.frequency = f
        time.sleep(delay)
        pow = pm.power
        assert isinstance(pow, float)
        return pow

    Pc, Pd = measure(c), measure(d)
    history = [(c, Pc), (d, Pd)]

    while abs(b - a) > tol:
        if Pc > Pd:
            b, Pd = d, Pc
            d = c
            c = a + invphi2 * (b - a)
            Pc = measure(c)
            history.append((c, Pc))
        else:
            a, Pc = c, Pd
            c = d
            d = a + invphi * (b - a)
            Pd = measure(d)
            history.append((d, Pd))
        if print_trace:
            print(f"a={a:.3e}, b={b:.3e}")
    # choose the best sampled point
    best_f, best_P = max(history, key=lambda x: x[1])
    rf_clock.frequency = best_f
    return best_f


def calc_segment_len_m(freq: float, c: float = 2e8) -> float:
    return c / freq


def calc_pulse_len(freq: float, seg_len: int, c: float = 2e8) -> float:
    return seg_len * calc_segment_len_m(freq, c=c)


def print_time_status(idx: int, t0: float, tot_iters: int):
    print(f"Done with iter {idx+1} of {tot_iters}")
    tot_time = time.time() - t0
    print(f"Total time so far: {(tot_time):.1f} s")
    average_time = tot_time / (idx + 1)
    iters_left = tot_iters - idx + 1
    time_left = average_time * iters_left
    end_time = (datetime.now() + timedelta(seconds=time_left)).strftime("%H:%M:%S")
    print(f"Estimated end time: {(end_time)}")


class DiodeControllerRouter:
    def __init__(self, controllers: list[EtekMLDC1032]):
        self.name_to_controller = {}
        for controller in controllers:
            assert isinstance(controller.edfas, dict)
            for edfa in controller.edfas:
                self.name_to_controller[edfa] = controller

    def __getattr__(self, method_name):
        def wrapper(edfa_name, *args, **kwargs):
            controller = self.name_to_controller.get(edfa_name)
            if not controller:
                raise ValueError(f"Unknown EDFA name: {edfa_name}")
            method = getattr(controller, method_name)
            return method(edfa_name, *args, **kwargs)

        return wrapper


def get_dark_currents(
    channels: list[int],
    scope: Agilent3000X,
    diode_controllers: list[EtekMLDC1032],
    scale_for_dark_current: float = 0.005,
) -> np.ndarray:
    dark_current_voltages = np.zeros(len(channels))
    turn_off_edfas(diode_controllers)
    for i, ch in enumerate(channels):
        prev_scale = scope.get_vertical_scale(ch)
        scope.set_vertical_scale(ch, scale_for_dark_current)
        scope.waveform.channel = ch
        scope.waveform.update_params()
        dark_current_voltages[i] = np.mean(scope.waveform.read_waveform()[1])
        scope.set_vertical_scale(ch, prev_scale)
        scope.waveform.update_params()
    turn_on_edfas(diode_controllers)
    return dark_current_voltages
