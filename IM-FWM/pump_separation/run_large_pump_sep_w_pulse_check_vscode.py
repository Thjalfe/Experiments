# %%
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys

sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
from laser_control import AndoLaser, TiSapphire
from verdi_laser import VerdiLaser
from amonics_edfa import EDFA
from picoscope2000 import PicoScope2000a
from thorlabs_mpc320 import ThorlabsMPC320, optimize_multiple_pol_cons
from arduino_pm import ArduinoADC

plt.ion()
import pyvisa as visa

rm = visa.ResourceManager()
print(rm.list_resources())
# %%
GPIB_val = 0
ando1_start = 1588 + 15
ando2_start = 1588 - 15
duty_cycle = [0.1, 0.2, 0.5, 1]
pulse_freq = 1 * 10**5
# edfa1 = EDFA("COM7")
arduino = ArduinoADC("COM11")
pol_con1 = ThorlabsMPC320(serial_no_idx=0)
pol_con2 = ThorlabsMPC320(serial_no_idx=1)# edfa2 = EDFA("COM5")
pico = PicoScope2000a()
pico.awg.set_square_wave_duty_cycle(pulse_freq, duty_cycle[1])
ando_pow_start = 0
ando1 = AndoLaser(ando1_start, GPIB_address=23)
ando2 = AndoLaser(ando2_start, GPIB_address=24)
time.sleep(0.1)
ando1.laser_on()
ando2.laser_on()
TiSa = TiSapphire(3)
verdi = VerdiLaser(com_port=4)
osa = OSA(
    ando2_start - 2,
    ando1_start + 2,
    resolution=0.05,
    GPIB_address=19,
    sweeptype="SGL",
)
EDFA_power = [2300]
data_folder = []
for pow in EDFA_power:
    data_folder.append(f"./data/CW/{pow}W/")


def start_edfa(edfa, edfa_pow, mode="APC"):
    if edfa is not None:
        edfa.set_mode(mode)
        time.sleep(0.1)
        edfa.set_status(1)
        time.sleep(0.1)
        edfa.set_current(edfa_pow)
        time.sleep(0.1)
        edfa.active_on()


def start_up(verdiPower, verdi, ando1, ando2):
    if verdi is not None:
        verdi.set_power(verdiPower)
        time.sleep(0.1)
        verdi.active_on()
        time.sleep(0.1)
        verdi.set_shutter(1)
    ando1.laser_on()
    ando2.laser_on()


# start_edfa(edfa1, EDFA_power[0])
# start_edfa(edfa2, 1150)
start_up(5, verdi, ando1, ando2)
# %%
# x, y = pol.brute_force_optimize(2, arduino, max_or_min="max", return_data=True, sleep_time=0.01)
optimize_multiple_pol_cons(
    arduino, pol_con1, pol_con2, max_or_min="max", interval=0.01, tolerance=0.5
)
# %%
mean_pump_wl = 1588
ando1_wl_lst = np.append(1589, np.arange(mean_pump_wl + 2.5, 1608 + 2.5, 2.5))
ando2_wl_lst = np.append(1587, np.arange(mean_pump_wl - 2.5, 1568 - 2.5, -2.5))
pump_sep_idx = 6
ando1_wl = ando1_wl_lst[pump_sep_idx]
ando1.set_wavelength(ando1_wl)
ando2_wl = ando2_wl_lst[pump_sep_idx]
ando2.set_wavelength(ando2_wl)
#%%
osa.span=(965.60,987.60)
osa.sweep()  # make sure osa.sweeptype="SGL"
wl = osa.wavelengths
power = osa.powers
spectrum = np.array([wl, power])
file_name = input("Enter a file name: ")
file_path = fr'C:\Users\FTNK-FOD\Desktop\Denis\measurments\{file_name}.npy'
np.save(file_path, spectrum)
osa.span=(1575,1605)
osa.sweep()  # make sure osa.sweeptype="SGL"
wlp = osa.wavelengths
powerp = osa.powers
spectrump = np.array([wlp, powerp])
file_name = input("Enter a file name: ")
file_path = fr'C:\Users\FTNK-FOD\Desktop\Denis\measurments\{file_name}.npy'
np.save(file_path, spectrump)

# %%


data_folder = "C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/large_pump_sep_within_L_band/pol_check/minimized"
file_name = f"pumps_{ando2_wl}_{ando1_wl}"
pulse_freq = 1 * 10**5
duty_cycle = [0.1, 0.2, 0.25, 0.5, 0.75, 1]
osa.set_res(0.05)
step_size = 0.1
num_steps = 1
# osa.set_sens("SHI1")
osa.set_sweeptype("SGL")
osa.sweep()
num_for_mean = 3
osa_data = {
    "pulse_freq": pulse_freq,
    "duty_cycle": duty_cycle,
    "wavelengths": np.zeros(
        (len(duty_cycle), num_steps, num_for_mean, len(osa.wavelengths))
    ),
    "powers": np.zeros((len(duty_cycle), num_steps, num_for_mean, len(osa.powers))),
}
t0 = time.time()
for k in range(num_steps):
    if k > 0:
        TiSa.delta_wl_nm(step_size)
    for i in range(len(duty_cycle)):
        print("Duty cycle: ", duty_cycle[i])
        for j in range(num_for_mean):
            pico.awg.set_square_wave_duty_cycle(pulse_freq, duty_cycle[i])
            osa.sweep()
            osa_data["wavelengths"][i, k, j, :] = osa.wavelengths
            osa_data["powers"][i, k, j, :] = osa.powers
t1 = time.time()
print(f"Time taken: {t1-t0:.2f} s")
# with open(f"{data_folder}/{file_name}_oscilloscope.pkl", "wb") as f:
#     pickle.dump(oscilloscope_data, f)
with open(f"{data_folder}/{file_name}_spectra.pkl", "wb") as f:
    pickle.dump(osa_data, f)
# |%%--%%| <BURpmFcmOb|Jnpn8nizq6>
# Stability of idler measurements
p1_wl = ando1.get_wavelength()
p2_wl = ando2.get_wavelength()
osa.set_res(0.01)
pico.awg.set_square_wave(pulse_freq, 0.5)
save_spec_dir = f"C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/C_plus_L_band/idler_stability_meas/after_mode_imaging/{p1_wl:.2f}nm_{p2_wl:.2f}nm"
if not os.path.exists(save_spec_dir):
    os.makedirs(save_spec_dir)
for i in range(50):
    osa.sweep()
    osa.save(f"{save_spec_dir}/sig_idler_{i}")
    print(i)
# |%%--%%| <Jnpn8nizq6|p0nkaUgozM>
ando1.set_wavelength(1608)
ando2.set_wavelength(1532)
# |%%--%%| <p0nkaUgozM|BURpmFcmOb>
p1_wl = ando1.get_wavelength()
p2_wl = ando2.get_wavelength()
osa.set_res(0.1)
pico.awg.set_square_wave(pulse_freq, 0.5)
save_spec_dir = f"C:/Users/FTNK-FOD/Desktop/Thjalfe/Experiments/IM-FWM/pump_separation/data/C_plus_L_band/after_pump_sep/5us_50percent/{p1_wl:.2f}nm_{p2_wl:.2f}nm"
if not os.path.exists(save_spec_dir):
    os.makedirs(save_spec_dir)
iter = 0
while True:
    TiSa.delta_wl_nm(0.1)
    osa.sweep()
    osa.save(f"{save_spec_dir}/spec_{iter}")
    iter += 1
    
# |%%--%%| <r0wTv6AXM7|8IzogKLM6d>
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))


