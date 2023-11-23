import numpy as np
import matplotlib.pyplot as plt
import sys
plt.ion()

sys.path.append(
    r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\InstrumentControl"
)
unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
from OSA_control import OSA
from laser_control import laser, TiSapphire
from verdi import verdiLaser

TiSa = TiSapphire(3)
verdi = verdiLaser(ioport="COM4")
osa = OSA(
    975,
    976,
    resolution=0.01,
    GPIB_num=[0, 19],
    sweeptype="SGL",
)
#|%%--%%| <zITb7XFN8J|zUe6uuPwPR>

#|%%--%%| <zUe6uuPwPR|ojtx735ZHv>
specs = np.load('specs_through_LPG.npy')
specs[specs < -80] = -80
num_specs = specs.shape[0]
fig, ax = plt.subplots()
for i in range(0, num_specs, 1):
    ax.scatter(specs[i, 0, :], specs[i, 1, :], linestyle='dotted', s=1)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Power (dBm)')
#|%%--%%| <ojtx735ZHv|GxmiSGiPLe>
# Statistics for the num_specs different spectra

