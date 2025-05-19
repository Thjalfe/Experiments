import numpy as np
import matplotlib.pyplot as plt

plt.style.use("custom")


x = 10  # pulse width in ns
y = 3000  # delay (gap) in ns

pulse_width = x * 1e-9
delay = y * 1e-9
period = pulse_width + delay

fs = 1e10  # sample rate in Hz (1 THz) to resolve ns features
num_periods = 10  # number of pulse periods to simulate
t_end = num_periods * period  # total time duration for simulation

t = np.linspace(0, t_end, int(fs * t_end), endpoint=False)

# and 0 during the delay. We use the modulus operator to get the repeating pattern.
signal = ((t % period) < pulse_width).astype(float)

N = len(t)
fft_signal = np.fft.fft(signal)
freq = np.fft.fftfreq(N, d=t[1] - t[0])

fft_signal_shifted = np.fft.fftshift(fft_signal)
freq_shifted = np.fft.fftshift(freq)
power_spectrum = np.abs(fft_signal_shifted) ** 2
epsilon = 1e-10
power_spectrum_dB = 10 * np.log10(power_spectrum + epsilon)

fig, ax = plt.subplots(2, 1)

ax[0].plot(t * 1e9, signal)  # converting time axis to ns for clarity
ax[0].set_title("Time Domain Signal")
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Amplitude")
ax[0].set_xlim(0, 5 * (x + y))  # zoom in to show a few periods

ax[1].plot(freq_shifted, power_spectrum_dB)
ax[1].set_title("Frequency Domain Spectrum")
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Magnitude")
ax[1].set_xlim([0, 1e9])
plt.show()
