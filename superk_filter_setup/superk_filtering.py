# %%
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import pickle


sys.path.append(r"C:\Users\FTNK-fod\Desktop\thjalfe\instrumentcontrol\LabInstruments")
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from osa_control import OSA
import importlib

plt.ion()
import pyvisa as visa


rm = visa.ResourceManager()
print(rm.list_resources())
#%%
osa = OSA(
    1200,
    2400,
    resolution=0.1,
    GPIB_address=17,
    sweeptype="RPT",
)
#%%
save_dir = "./data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
old_res = osa.resolution
osa.span=1200,2400
osa.resolution = 0.5
osa.sweeptype = "SGL"
osa.sensitivity="SHI1"
osa.sweep()
filename = f"pow=50per_FELH1500_res={osa.resolution}_SHI1.txt"
filename = f"pow=50per_FELH1500_FESH1000_res={osa.resolution}_SHI1_span={osa.span[0]:.1f}-{osa.span[1]:.1f}.txt"
#filename = f"pow=50per_ref_res={osa.resolution}.txt"
osa.sweeptype="RPT"
osa.resolution = old_res
spectrum = np.vstack((osa.wavelengths, osa.powers))
np.savetxt(os.path.join(save_dir, filename), spectrum)
osa.sweep()