import numpy as np
import matplotlib.pyplot as plt
from util_funcs import load_pump_files
pump_folder = './data/char_setup/pump_powers/'
pump_files = ['throughsetup', 'wdm_tap']
through_setup = load_pump_files(pump_folder, pump_name=pump_files[0])
wdm_tap = load_pump_files(pump_folder, pump_name=pump_files[1])
fig, ax = plt.subplots(2, 1)
for idx, pumps in enumerate(through_setup[1]):
    ax[0].plot(through_setup[0][idx, :, 0], through_setup[0][idx, :, 1], label=f'{pumps}')
    ax[1].plot(wdm_tap[0][idx, :, 0], wdm_tap[0][idx, :, 1], label=f'{pumps}')
# ax[0].legend()
# ax[1].legend()
plt.show()
#|%%--%%| <1VOFSJRQM2|mjIkkY37d2>
from scipy.signal import find_peaks

# |%%--%%| <mjIkkY37d2|j2HPwWZDlL>
# ref: outside setup , throughsetup_doublemmf:
ref_tisa = np.loadtxt('./data/char_setup/tisa/TiSa_directly_outsidesetup_res0.2nm.csv', delimiter=',')
# throughsetup: normal setup with MMF pickup
tisa_throughsetup = np.loadtxt('./data/char_setup/tisa/TiSa_through_setup_res0.2nm.csv', delimiter=',')
# throughsetup: normal setup with 2xMMF pickup
tisa_throughsetup_doublemmf = np.loadtxt('./data/char_setup/tisa/TiSa_through_setup_doubleMMF_res0.2nm.csv', delimiter=',')
# through setup without freespace pickup, but attenuation directly on osa
tisa_no_freespace = np.loadtxt('./data/char_setup/tisa/TiSa_no_freespace_res0.2nm.csv', delimiter=',')
# through setup without freespace pickup, with free space attenuation on tisa
tisa_no_freespace_noatten = np.loadtxt('./data/char_setup/tisa/TiSa_no_freespace_noattenuation_res0.2nm.csv', delimiter=',')
# through setup without freespace pickup, with polarization attenuation on tisa
tisa_no_freespace_noatten2 = np.loadtxt('./data/char_setup/tisa/TiSa_no_freespace_noattenuation_res0.2nm_1.csv', delimiter=',')
# through setup with SMF pickup fiber
tisa_SMF = np.loadtxt('./data/char_setup/tisa/TiSa_freespace_SMF_res0.2nm.csv', delimiter=',')
tisa_SMF2 = np.loadtxt('./data/char_setup/tisa/TiSa_freespace_SMF_res0.2nm_1.csv', delimiter=',')
tisa_SMF3 = np.loadtxt('./data/char_setup/tisa/TiSa_freespace_SMF_res0.2nm_1_2.csv', delimiter=',')
tisa_SMF4 = np.loadtxt('./data/char_setup/tisa/TiSa_freespace_SMF_res0.2nm_1_2_3.csv', delimiter=',')
tisa_SMF_1060 = np.loadtxt('./data/char_setup/tisa/TiSa_freespace_SMF_1060_res0.2nm.csv', delimiter=',')
tisa_SMF_1060_2 = np.loadtxt('./data/char_setup/tisa/TiSa_freespace_SMF_1060_res0.2nm_1.csv', delimiter=',')
time_array = np.linspace(0, 50, len(ref_tisa[:, 0]))
plt.plot(time_array, ref_tisa[:, 1], label='Outside setup')
# plt.plot(time_array, tisa_throughsetup[:, 1], label='Through setup')
plt.plot(time_array, tisa_throughsetup_doublemmf[:, 1], label='Through setup with double MMF')
# plt.plot(time_array, tisa_no_freespace[:, 1], label='Through setup without freespace')
# plt.plot(time_array, tisa_no_freespace_noatten[:, 1], label='Through setup without freespace and attenuation')
plt.plot(time_array, tisa_no_freespace_noatten2[:, 1], label='Through setup without freespace and attenuation 2')
# plt.plot(time_array, tisa_SMF[:, 1], label='Through setup with SMF')
# plt.plot(time_array, tisa_SMF2[:, 1], label='Through setup with SMF 2')
plt.plot(time_array, tisa_SMF3[:, 1] - 3, label='Through setup with SMF 3')
plt.plot(time_array, tisa_SMF4[:, 1] - 3, label='Through setup with SMF 4')
plt.plot(time_array, tisa_SMF_1060[:, 1] - 3, label='Through setup with HI 1060')
plt.plot(time_array, tisa_SMF_1060_2[:, 1] - 3, label='Through setup with HI 1060 2')
plt.title('TiSa through setup')
plt.xlabel('Time (s)')
plt.ylabel('Power (dBm)')
plt.legend()
# plt.savefig('./figs/char_setup/TiSa_through_setup.pdf', bbox_inches='tight')
