import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal import firwin  
from scipy.fft import fft, fftfreq
import sys
import scipy.signal  # to use sawtooth, square
import sounddevice as s
from scipy.signal import resample
from scipy.io.wavfile import write


def signal_components(amplitude, frequency, duration):

    components = [amplitude, frequency, duration]
    return components

# This creates a signal
def createSignal(waveform, time_domain, signalComponents):
    
    if (waveform.lower() == "sine" ):
        signal = signalComponents[0] * np.sin(2 * np.pi * signalComponents[1] * time_domain)
        
    elif waveform.lower() == "saw":
        signal = signalComponents[0] * scipy.signal.sawtooth(2 * np.pi * signalComponents[1] * time_domain)

    elif (waveform.lower() == "square"):
        signal = signalComponents[0] * scipy.signal.square(2 * np.pi * signalComponents[1] * time_domain)

    return signal

# This trasnfers the time domain signal into the frequency domain using fft
def frequencyDomain_transfer(signal):
    
    fft_sig = fft(signal)
    mags = np.abs(fft_sig)

    return mags

def make_frequency_bins(signal): # This makes "bins" to put the different frequyency values in 
    # https://numpy.org/doc/2.2/reference/generated/numpy.fft.fftfreq.html
    # https://dsp.stackexchange.com/questions/26927/what-is-a-frequency-bin 

    freq_bins = fftfreq(len(signal), d=1/44100)
    return freq_bins

# This plots the graph 
def plotGraph(mags, freqBins, signal):


    fig, (plt1, plt2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Plot Time Domain
    timeDomain = np.linspace(0, 10, int(44100 * 10))
    plt1.plot(timeDomain, signal)
    plt1.set_xlim(0,1.0)
    plt1.set_title(f"Time Domain")
    plt1.set_xlabel("Duration (secs)")
    plt1.set_ylabel("Amplitude")

    # Plot frequency domain
    plt2.plot(freqBins, mags)
    plt2.set_xlim(left=0, right=max(freqBins))
    plt2.set_xlabel("Frequencies (Hz)")
    plt2.set_ylabel("Amplitude (magnitude)")
    plt2.set_title("Frequency Domain)")
    fig.tight_layout() 

def sum_signals(signal1, signal2, signal3):
    
    sum = []

    for i in range(len(signal1)):
        sum.append( signal1[i] + signal2[i] + signal3[i])

    return sum


timeDomain = np.linspace(0, 10, int(44100 * 10))

comps = signal_components(0.04, 261.63, 10)
C4_Time = createSignal('Saw', timeDomain, comps)
C4_FreqMags = frequencyDomain_transfer(C4_Time)
C4_FreqBins = make_frequency_bins(C4_Time)

comps = signal_components(0.04, 329.63, 10)
E4_Time = createSignal('Saw', timeDomain, comps)
E4_FreqMags = frequencyDomain_transfer(E4_Time)
E4_FreqBins = make_frequency_bins(E4_Time)

#comps = signal_components(0.04, 392, 10)
comps = signal_components(0.04, 196, 10)

G4_Time = createSignal('Saw', timeDomain, comps)
G4_FreqMags = frequencyDomain_transfer(G4_Time)
G4_FreqBins = make_frequency_bins(G4_Time)

CMajor_Time = sum_signals(C4_Time, E4_Time, G4_Time)
CMajorMags = frequencyDomain_transfer(CMajor_Time)
CMajor_FreqBins = make_frequency_bins(CMajor_Time)
plotGraph(CMajorMags, CMajor_FreqBins, CMajor_Time)

# plt.axvline(x=261.63, color='r', linestyle='--') # C4
# plt.axvline(x=329.63, color='r', linestyle='--') # E4
# plt.axvline(x=392, color='r', linestyle='--') # G4

plt.show()

# # # Resources: 
# # #   https://matplotlib.org/stable/index.html
# # #   https://docs.scipy.org/doc/scipy/
# # #   https://numpy.org/doc/2.3/user/index.html#user
# # #   https://stackoverflow.com/questions/543309/programmatically-stop-execution-of-python-script
# # #   https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
# # #   https://www.dspguide.com/ch3/2.htm
# # #   https://pysdr.org/content/sampling.html
# # #   https://dsp.stackexchange.com/questions/26927/what-is-a-frequency-bin
# # # https://pysdr.org/content/filters.html
# # https://docs.scipy.org/doc/scipy/reference/signal.html