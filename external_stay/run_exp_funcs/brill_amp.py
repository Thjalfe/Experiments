from dataclasses import dataclass, field
import pickle
import numpy as np
from external_stay.run_exp_funcs.helper_funcs import (
    increment_segment,
    get_n_scope_traces,
    get_osa_spec,
)
from clients.osa_clients import OSAClient
from clients.oscilloscope_clients import HP83480
from clients.diode_controller_clients import EtekMLDC1032
from clients.hp81200_client import HP81200Client
from external_stay.processing.processing_funcs.brill_amp import calc_gain


@dataclass
class TracesSpectraMultipleColls:
    num_reps_for_traces: int
    dummy_trace_shape: tuple[int, ...]
    dummy_osa_spec_shape: tuple[int, ...]
    num_segment_steps: int

    pump_ref_time_traces: np.ndarray = field(init=False)

    signal_ref_time_traces: np.ndarray = field(init=False)

    pump_w_amplification_time_traces: np.ndarray = field(init=False)

    signal_w_amplification_spectrum: np.ndarray = field(init=False)
    signal_w_amplification_time_traces: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        t0, t1 = self.dummy_trace_shape
        s0, s1 = self.dummy_osa_spec_shape

        self.pump_ref_spectrum = np.zeros(self.dummy_osa_spec_shape)
        self.pump_ref_time_traces = np.zeros((self.num_reps_for_traces, t0, t1))

        self.signal_ref_spectrum = np.zeros(self.dummy_osa_spec_shape)
        self.signal_ref_time_traces = np.zeros((self.num_reps_for_traces, t0, t1))

        self.pump_w_amplification_time_traces = np.zeros(
            (self.num_segment_steps, self.num_reps_for_traces, t0, t1)
        )

        self.signal_w_amplification_spectrum = np.zeros(
            (self.num_segment_steps, s0, s1)
        )

        self.signal_w_amplification_time_traces = np.zeros(
            (self.num_segment_steps, self.num_reps_for_traces, t0, t1)
        )

    def calc_gain(self) -> None:
        gain_vs_segment_step = []
        for i in range(self.num_segment_steps):
            gain, sig_power = calc_gain(
                self.wl_ax,
                self.signal_ref_spectrum,
                self.pump_ref_spectrum,
                self.signal_w_amplification_spectrum[i],
            )
            gain_vs_segment_step.append(gain)
        self.gain_vs_segment_step = gain_vs_segment_step

    @staticmethod
    def _compute_mean_std_norm(
        arr: np.ndarray, axis: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(arr, axis=axis)
        std = np.std(arr, axis=axis)
        norm_factor = np.max(mean)
        return mean, std, mean / norm_factor, std / norm_factor

    def mean_std_traces(self) -> None:
        (
            self.mean_pump_ref,
            self.std_pump_ref,
            self.mean_pump_ref_norm,
            self.std_pump_ref_norm,
        ) = self._compute_mean_std_norm(self.pump_ref_time_traces[:, 1, :], axis=0)

        self.mean_pump, self.std_pump, self.mean_pump_norm, self.std_pump_norm = (
            self._compute_mean_std_norm(
                self.pump_w_amplification_time_traces[:, :, 1, :]
            )
        )

        (
            self.mean_sig_ref,
            self.std_sig_ref,
            self.mean_sig_ref_norm,
            self.std_sig_ref_norm,
        ) = self._compute_mean_std_norm(self.signal_ref_time_traces[:, 1, :], axis=0)

        (
            self.mean_signal,
            self.std_signal,
            self.mean_signal_norm,
            self.std_signal_norm,
        ) = self._compute_mean_std_norm(
            self.signal_w_amplification_time_traces[:, :, 1, :]
        )

    def to_dict(self) -> dict:
        return {
            "num_reps_for_traces": self.num_reps_for_traces,
            "dummy_trace_shape": self.dummy_trace_shape,
            "dummy_osa_spec_shape": self.dummy_osa_spec_shape,
            "num_segment_steps": self.num_segment_steps,
            "pump_ref_spectrum": self.pump_ref_spectrum,
            "pump_ref_time_traces": self.pump_ref_time_traces,
            "signal_ref_spectrum": self.signal_ref_spectrum,
            "signal_ref_time_traces": self.signal_ref_time_traces,
            "pump_w_amplification_time_traces": self.pump_w_amplification_time_traces,
            "signal_w_amplification_spectrum": self.signal_w_amplification_spectrum,
            "signal_w_amplification_time_traces": self.signal_w_amplification_time_traces,
        }

    @staticmethod
    def from_dict(data: dict) -> "TracesSpectraMultipleColls":
        instance = TracesSpectraMultipleColls(
            data["num_reps_for_traces"],
            data["dummy_trace_shape"],
            data["dummy_osa_spec_shape"],
            data["num_segment_steps"],
        )
        instance.pump_ref_spectrum = data.get("pump_ref_spectrum", None)
        instance.pump_ref_time_traces = data["pump_ref_time_traces"]
        instance.signal_ref_spectrum = data.get("signal_ref_spectrum", None)
        instance.signal_ref_time_traces = data["signal_ref_time_traces"]
        instance.pump_w_amplification_time_traces = data[
            "pump_w_amplification_time_traces"
        ]
        instance.signal_w_amplification_spectrum = data[
            "signal_w_amplification_spectrum"
        ]
        instance.signal_w_amplification_time_traces = data[
            "signal_w_amplification_time_traces"
        ]
        return instance

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @staticmethod
    def load(file_path: str) -> "TracesSpectraMultipleColls":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return TracesSpectraMultipleColls.from_dict(data)

    def traces_and_spectra_multiple_collisions_meas(
        self,
        scope: HP83480,
        osa: OSAClient,
        diode_contr_1: EtekMLDC1032,
        diode_contr_2: EtekMLDC1032,
        hp81200: HP81200Client,
        signal_scope_position: float,
        pump_scope_position: float,
        segment_stepsize: int,
        module_id_to_move: int = 1,
        segment_stepsize_time: float = 1e-9,
        segment_direction: int = -1,
        signal_channel: int = 1,
        pump_channel: int = 3,
    ) -> None:
        # TODO: Make method of above dataclass
        def take_multiple_scope_meas_for_both_pulses(
            scope: HP83480,
            signal_scope_position: float,
            pump_scope_position: float,
            num_reps_for_traces: int,
            signal_channel: int,
            pump_channel: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            scope.timebase.position = pump_scope_position
            pump_traces = get_n_scope_traces(scope, num_reps_for_traces, pump_channel)
            scope.timebase.position = signal_scope_position
            sig_traces = get_n_scope_traces(scope, num_reps_for_traces, signal_channel)
            return sig_traces, pump_traces

        self.stepsize_time = segment_stepsize * segment_stepsize_time
        osa.sweeptype = "SGL"
        osa.stop_sweep()
        num_reps_for_traces = self.num_reps_for_traces
        num_segment_steps = self.num_segment_steps
        # Starting with amplification
        diode_contr_1.set_edfa_to_max("PRE2_1")
        diode_contr_2.set_edfa_to_max("PRE2_2")
        diode_contr_1.set_edfa_to_max("PRE1")
        diode_contr_1.set_edfa_to_max("EDFA2")
        diode_contr_2.set_edfa_to_max("EDFA1")
        for i in range(num_segment_steps):
            if i != 0:
                increment_segment(
                    hp81200, module_id_to_move, segment_direction * segment_stepsize
                )
                pump_scope_position = (
                    pump_scope_position
                    + segment_direction * segment_stepsize * segment_stepsize_time
                )
            osa.sweep()
            spectrum = get_osa_spec(osa)
            signal_traces, pump_traces = take_multiple_scope_meas_for_both_pulses(
                scope,
                signal_scope_position,
                pump_scope_position,
                num_reps_for_traces,
                signal_channel,
                pump_channel,
            )
            self.signal_w_amplification_spectrum[i] = spectrum
            self.signal_w_amplification_time_traces[i] = signal_traces
            self.pump_w_amplification_time_traces[i] = pump_traces
            print(f"Moved {i+1} out of num_reps_for_traces")
        scope.timebase.position = pump_scope_position
        diode_contr_1.disable_edfa("EDFA2")
        traces = get_n_scope_traces(scope, num_reps_for_traces, pump_channel)
        osa.sweep()
        spectrum = get_osa_spec(osa)
        self.pump_ref_time_traces = traces
        self.pump_ref_spectrum = spectrum
        scope.timebase.position = signal_scope_position
        diode_contr_1.set_edfa_to_max("EDFA2")
        diode_contr_1.disable_edfa("PRE1")
        diode_contr_2.disable_edfa("EDFA1")
        traces = get_n_scope_traces(scope, num_reps_for_traces, pump_channel)
        osa.sweep()
        spectrum = get_osa_spec(osa)
        self.signal_ref_time_traces = traces
        self.signal_ref_spectrum = spectrum
        self.time_ax = traces[0]
        self.wl_ax = spectrum[0]
        osa.sweeptype = "RPT"
        osa.sweep()
