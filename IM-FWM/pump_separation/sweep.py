# %%
import numpy as np
from typing import List
import os
import time
import matplotlib.pyplot as plt
import sys
import importlib

sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser, TiSapphire, PhotoneticsLaser
from verdi_laser import VerdiLaser
from picoscope2000 import PicoScope2000a
from pol_cons import ThorlabsMPC320
from arduino_pm import ArduinoADC

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
pump1_start = 1608
pump2_start = 1604
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0, initial_velocity=50)
pol_con2 = ThorlabsMPC320(serial_no_idx=1, initial_velocity=50)
pico = PicoScope2000a()
pulse_freq = 10**5
pico.awg.set_square_wave_duty_cycle(pulse_freq, 0.2)
ando_pow_start = 0
# pump1_laser = PhotoneticsLaser(pump1_start, GPIB_address=10, power=6)
pump1_laser = AndoLaser(pump1_start, GPIB_address=23, power=8)
pump2_laser = AndoLaser(pump2_start, GPIB_address=24, power=8)
time.sleep(0.1)
pump1_laser.enable()
pump2_laser.enable()
tisa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    960,
    990,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="RPT",
    trace="a",
)

# verdi.power = 6
# verdi.shutter = 0
time.sleep(0.5)
# %%
p1_wl = pump1_laser.wavelength
p2_wl = pump1_laser.wavelength
data_dir = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\L_bandpumps_phasematch_around_QDotWL"
osa.stop_sweep()
osa.sweeptype = "SGL"
for i in range(30):
    t0 = time.time()
    tisa.delta_wl_nm(0.1)
    osa.sweep(update_spectrum=False)
    print(f"Time taken for sweep: {time.time()-t0:.2f} s")

    # spectrum = np.vstack((osa.wavelengths, osa.powers))
    # np.savetxt(
    #     os.path.join(
    #         data_dir,
    #         f"spec_p2wl={pump2_laser.wavelength:.1f}_p1pm_wl={p1_wl:.2f}_iter={i}.csv",
    #     ),
    #     spectrum,
    #     fmt="%f",
    #     delimiter=",",
    # )
# %%
# Check different duty cycle CE experiment
import pickle

data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/pulse/10us_cycle/"
file_name = "pumps_1568_1578"
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]

osa.sweep()
oscilloscope_data = {
    "pulse_freq": pulse_freq,
    "duty_cycle": duty_cycle,
    "time_ref": np.zeros((len(duty_cycle), 1250)),
    "voltage_ref": np.zeros((len(duty_cycle), 1250)),
    "time_sig": np.zeros((len(duty_cycle), 1250)),
    "voltage_sig": np.zeros((len(duty_cycle), 1250)),
}
osa.get_spectrum()
osa_data = {
    "pulse_freq": pulse_freq,
    "duty_cycle": duty_cycle,
    "wavelengths": np.zeros((len(duty_cycle), len(osa.wavelengths))),
    "powers": np.zeros((len(duty_cycle), len(osa.powers))),
}
for i in range(len(duty_cycle)):
    pico.awg.set_square_wave(pulse_freq, duty_cycle[i])
    pico.oscilloscope.set_params(
        channel=2,
        enabled=0,
    )
    pico.oscilloscope.set_params(
        channel=0,
        enabled=1,
        channel_range=7,
        oversample=2,
        timebase=3,
        threshold=50,
    )
    pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
    time_axis_ref, y_ref, overflow, adc1, pulse_length, pulse_dist = (
        pico.oscilloscope.run_scope(threshold_high=0.5, threshold_low=0.5)
    )
    pico.oscilloscope.set_params(
        channel=0,
        enabled=0,
    )
    range_list = [7, 7, 7, 5, 5, 5]
    oscilloscope_range = range_list[i]
    pico.oscilloscope.set_params(
        channel=2,
        enabled=1,
        channel_range=oscilloscope_range,
        oversample=2,
        timebase=3,
        threshold=50,
    )
    pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
    time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = (
        pico.oscilloscope.run_scope()
    )
    oscilloscope_data["time_ref"][i, :] = time_axis_ref
    oscilloscope_data["voltage_ref"][i, :] = y_ref
    oscilloscope_data["time_sig"][i, :] = time_axis
    oscilloscope_data["voltage_sig"][i, :] = y
    osa.sweep()
    osa_data["wavelengths"][i, :] = osa.wavelengths
    osa_data["powers"][i, :] = osa.powers
with open(f"{data_folder}/{file_name}_oscilloscope.pkl", "wb") as f:
    pickle.dump(oscilloscope_data, f)
with open(f"{data_folder}/{file_name}_spectra.pkl", "wb") as f:
    pickle.dump(osa_data, f)
# %%
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]
pico.awg.set_square_wave(pulse_freq, duty_cycle[-2])
pico.oscilloscope.set_params(
    channel=2,
    enabled=0,
)
pico.oscilloscope.set_params(
    channel=0,
    enabled=1,
    channel_range=7,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
time_axis1, y1, overflow, adc1, pulse_length, pulse_dist = pico.oscilloscope.run_scope(
    threshold_high=0.5, threshold_low=0.5
)
pico.oscilloscope.set_params(
    channel=0,
    enabled=0,
)
# [7, 7, 7, 5, 5, 5]
pico.oscilloscope.set_params(
    channel=2,
    enabled=1,
    channel_range=7,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico.oscilloscope.set_sampling_rate(pulse_freq, peaks_wanted=2)
time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = (
    pico.oscilloscope.run_scope()
)
# fig, ax = plt.subplots(1, 1)
# ax.plot(time_axis1, y1)
# ax1 = ax.twinx()
# ax1.plot(time_axis, y, color='red')
# plt.show()
# %%
pico.oscilloscope.set_params(
    channel=3,
    enabled=1,
    channel_range=2,
    oversample=2,
    timebase=3,
    threshold=50,
)
pico.oscilloscope.set_sampling_rate(20000, peaks_wanted=1)
while True:
    time_axis, y, overflow, adc, pulse_length_out, pulse_dist_out = (
        pico.oscilloscope.run_scope()
    )
    print(np.mean(y))
    time.sleep(0.1)
