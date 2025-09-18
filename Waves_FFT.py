import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# https://matplotlib.org/stable/tutorials/pyplot.html
# https://pysdr.org/content/frequency_domain.html#fft-in-python
# https://docs.vultr.com/python/third-party/numpy/sin # SIN
# https://stackoverflow.com/questions/66000468/plot-square-wave-in-python SQUARE WAVE


# THE FOLLOWING IS FROM PYSDR TEXTBOOK FREQUENCY DOMAIN SECTION (annotated by me)

# This returns an array of ints from 0 to n - 1, steps of 1
# for example, t = [0,1,2, ..., 99]

#This is our time parameter / where in the sine wave
t = np.arange(50)

# This an array of the values computed at each single point in t\
# s stands for signal, will output values between -1 and +1``

# For example t[1] = 1 is 0.15 * 2pi * 1 = .9425, so sin(0.9425) = 0.8090
# Then, t[2] = 2 * 0.9425 = 1.8850, so sin(1.8850) = .9460
# then, t[3] = 3 * 0.9425 = 2.875, so sin(2.8275) = 0.3090
# then, t[4] = 4 * 0.9425 = 3.7700, so sin(3.7700) = -0.5878
#  and so on... (The final values in each line are plotted)

# SIN WAVE

#0.15 is frequency/cycle * 2pi * t (think of as seconds)
# s = np.sin(0.10*2*np.pi*t)
# plt.plot(t, s, 'r-')
# plt.show()


#Perform Fast Fourier Transform
# S = np.fft.fft(s) # Converts S into an array of complex numbers via fast fourier transform

# # Make more sense of them via calculating magnitude and phase
# S_mag = np.abs(S) # How Strong each frequency component is
# S_phase = np.angle(S) # The timing of each frequency component

# # Perform an fft shift, to get 0 hz in center and negative frequenceis to the left
# S = np.fft.fftshift(np.fft.fft(S))

# # Plot the magnitude 
# plt.figure(1)
# plt.plot(t, S_mag, 'r-')
# plt.xlabel("FFT Index")
# plt.ylabel("FFT Magnitde")

# # Plot the Phase
# plt.figure(2)
# plt.plot(t,S_phase, '.-')
# plt.xlabel("FFT Index")
# plt.ylabel("FFT Phase [radians]")

# #Show both
# plt.grid()
# plt.show()

# SQUARE WAVE (This was taken from stack overflow as referenced at top documentation)

# sq = np.linspace(0, 1, 500, endpoint=False)
# plt.plot(t, signal.square(2 * np.pi * 5 * t),'b')
# plt.ylim(-2, 2)
# plt.grid()
# plt.show()


# THE FOLLOWING IS MY OWN EXPERIMENTATION AND KNOWLEDGE (with using sci.py)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
from scipy.signal import periodogram

t = np.linspace(0, 1, 1000, endpoint=False)
saw = signal.sawtooth(2 * np.pi * 5 * t)

saw_fft = np.fft.fft(saw) # Converts saw signal from time domain to the frequency domain/
# shift to 0 hz
saw_fft_shifted = np.fft.fftshift(saw_fft)

saw_mag = np.abs(saw_fft_shifted) # How Strong each frequency component is

# frequency domain mean that is shows what frequencies are presnt in the signal and relative strenghts
freq = np.fft.fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))

plt.figure(1)
plt.plot(freq, saw_mag)
plt.xlabel('Frequecy in HZ')
plt.ylabel('Magnitude')
plt.grid()

square = signal.square(2 * np.pi * 5 * t)

square_fft = np.fft.fft(square) # Converts saw signal from time domain to the frequency domain/
# shift to 0 hz
square_fft_shifted = np.fft.fftshift(square_fft)

square_mag = np.abs(square_fft_shifted) # How Strong each frequency component is

# frequency domain mean that is shows what frequencies are presnt in the signal and relative strenghts
freq = np.fft.fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))
plt.figure(2)

plt.plot(freq, square_mag)
plt.xlabel('Frequecy in HZ')
plt.ylabel('Magnitude')
plt.grid()
