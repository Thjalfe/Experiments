import numpy as np
import json
from dataclasses import dataclass


def rolling_average(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode="same")


def dBm_to_mW(dBm: np.ndarray | float) -> np.ndarray | float:
    return 10 ** (dBm / 10)


def mW_to_dBm(mW: np.ndarray | float) -> np.ndarray | float:
    return 10 * np.log10(mW)


@dataclass
class Fiber:
    full_fiber_name: str
    length: float
    insertion_loss: float


def load_fibers_from_json(
    file_path: str = "/home/thjalfe/Documents/PhD/Projects/Experiments/external_stay/data/fibers.json",
) -> dict[str, Fiber]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return {name: Fiber(**params) for name, params in data.items()}


def get_fiber(fiber_name: str, fiber_dict: dict) -> Fiber:
    try:
        return fiber_dict[fiber_name]
    except KeyError:
        raise ValueError(
            f"Fiber '{fiber_name}' not found. Available fibers: {list(fiber_dict.keys())}"
        )
