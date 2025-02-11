import logging
from typing import List
import os
from laser_control import Laser
from verdi_laser import VerdiLaser


class CustomErrorHandler(logging.Handler):
    def __init__(
        self, lasers: List[Laser], verdi: VerdiLaser, amplifiers: List[object]
    ):
        super().__init__()
        self.lasers = lasers
        self.verdi = verdi
        self.amplifiers = amplifiers

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            turn_off_all_lasers(self.lasers, self.verdi, self.amplifiers)


def turn_off_all_lasers(
    lasers: List[Laser],
    verdi: VerdiLaser,
    amplifiers: List[object],
):
    for laser in lasers:
        laser.disable()
    verdi.shutter = 0
    for amp in amplifiers:
        amp.shutter = 0


def setup_logging(
    data_folder: str,
    filename: str,
    lasers: List[Laser],
    verdi: VerdiLaser,
    amplifiers: List[object],
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(os.path.join(data_folder, filename), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Stream handler to log messages to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # error handler
    custom_error_handler = CustomErrorHandler(lasers, verdi, amplifiers)
    custom_error_handler.setFormatter(formatter)
    logger.addHandler(custom_error_handler)
    return logger


def logging_message(logger: logging.Logger, message: str, level: str = "info"):
    log_method = getattr(logger, level)
    log_method(message)
