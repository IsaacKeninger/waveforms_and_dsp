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


# Chat GPT Prompt: codebase + how to make each signal the same length no matter what" made this function
def resample_to_fixed_length(signal, target_length=44100):
    return resample(signal, target_length)

def signal_components(amplitude, frequency, duration):

    components = [amplitude, frequency, duration]
    return components

# Calculates The Nyquist Rate, 2x the Frequency (in reality it is the highest frequency, but I am using constant ones to demonstrate)
def getNyquistRate(frequency):
    return frequency * 2

# Makes the time domain, x axis depending on the sample rate and duration
def makeTimeDomain(duration, sample_rate):

    timeDomain = np.linspace(0, duration, int(sample_rate * duration))
    return timeDomain

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

def make_frequency_bins(signal, sample_rate): # This makes "bins" to put the different frequyency values in 
    # https://numpy.org/doc/2.2/reference/generated/numpy.fft.fftfreq.html
    # https://dsp.stackexchange.com/questions/26927/what-is-a-frequency-bin 

    freq_bins = fftfreq(len(signal), d=1/sample_rate)
    return freq_bins

# This plots the graph 
def plotGraph(freqBins, mags, time_domain, signal, sampleRateType, cutoff=0):

    fig, (plt1, plt2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Plot Time Domain
    plt1.plot(time_domain, signal)
    plt1.set_xlim(0,1.0)
    plt1.set_title(f"Time Domain: {sampleRateType}")
    plt1.set_xlabel("Duration (secs)")
    plt1.set_ylabel("Amplitude")

    # Plot frequency domain
    plt2.plot(freqBins, mags)
    plt2.axvline(x=cutoff, color='red', linestyle='--')
    plt2.set_xlim(left=0, right=max(freqBins))
    plt2.set_xlabel("Frequencies (Hz)")
    plt2.set_ylabel("Amplitude (magnitude)")
    plt2.set_title(f"Frequency Domain: {sampleRateType}")
    fig.tight_layout() 

def lowpassFilter(signal, fs,cutoffFreq):
    num_tap = 91 # Number of filter coefficents, more taps = a sharper filter. 

    lowPassFilter = scipy.signal.firwin(num_tap, cutoffFreq, fs=fs)  # FIR means finite impuse response. this looks at number of samples to calculatee output samples

    filtered = scipy.signal.lfilter(lowPassFilter,[1.0],signal) # This applies FIR to my sample. Weights, denominator coefficents. Takes each sample, looks at current and past, multipleies them by fir coefficents, sums them up
    print(filtered)
    return filtered

def play(signal):
    print("Playing...")
    s.play(signal)
    s.wait()

def Analyze(amplitude, frequency, duration, waveform, sampleRateType, filter):

# Get specific signal components
    signalParts = signal_components(amplitude, frequency, duration)

    if sampleRateType.lower() == 'undersampled':
        sampleRate = getNyquistRate(signalParts[1]) / 2
    elif (sampleRateType.lower() == 'proper'):
        sampleRate = getNyquistRate(signalParts[1]) * 10
    elif (sampleRateType.lower() == 'oversampled'):
        sampleRate = getNyquistRate(signalParts[1]) * 50
    elif (sampleRateType.lower() == "nyquist"):
        sampleRate = getNyquistRate(signalParts[1])
    else:
        sys.exit("SampleRateType must be one of the following (Undersampled, Proper, Oversampled, Nyquist).") # This stops the program
        
    timeDomain = makeTimeDomain(signalParts[2], sampleRate)

    # make signal
    signal = createSignal(waveform, timeDomain, signalParts)

   # This resamples the sample so they can be the same length
    signal = resample_to_fixed_length(signal, target_length=44100)
    effective_fs = len(signal) / signalParts[2]
    timeDomain = np.linspace(0, duration, len(signal))

    if (filter == 'y'):
        cutoffFreq = signalParts[1] * 1.5  
        signal = lowpassFilter(signal,effective_fs, cutoffFreq)

    # Convert Signal to frequency domain and track magnitudes at each freuqnecy
    signal_magnitudes = frequencyDomain_transfer(signal)

    # Make section domain for certain frequenceis
    frequnecy_bins= make_frequency_bins(signal, effective_fs)
    
    play(signal)
    s.wait()

    # PLOT GRAPH
    if (filter == 'y'):
        plotGraph(frequnecy_bins, signal_magnitudes, timeDomain, signal, sampleRateType, cutoffFreq)
    else:
        plotGraph(frequnecy_bins, signal_magnitudes, timeDomain, signal, sampleRateType)

# Sin waves
# # Analyze(1.0, 25.0, 1.0, 'sin', 'Undersampled')
# Analyze(1.0, 25.0, 1.0, 'sin', 'Proper', 'n')
# Analyze(1.0, 25.0, 1.0, 'sin', 'Oversampled', 'y')
# # Analyze(1.0, 25.0, 1.0, 'sin', 'Nyquist')
# Analyze(0.04, 261, 10, 'saw', 'Proper','n')
# Analyze(0.04, 261, 10, 'saw', 'Proper', 'y')

# # # # # Saw Waves
# Analyze(0.04, 50, 10, 'saw', 'Proper','n')

# Analyze(0.04, 50, 10, 'saw', 'Proper', 'y')
# Analyze(1.0, 25.0, 1.0, 'saw', 'Oversampled','n')
# # Analyze(1.0, 25.0, 1.0, 'saw', 'Nyquist','n')

# # Square Waves
Analyze(0.04, 25, 10, 'square', 'Proper','n')
Analyze(0.04, 25, 10, 'square', 'Proper', 'y')
# Analyze(1.0, 25.0, 1.0, 'square', 'Undersampled')
# Analyze(1.0, 25.0, 1.0, 'square', 'Proper','n')
# Analyze(1.0, 25.0, 1.0, 'square', 'Proper','y')
# Analyze(1.0, 25.0, 1.0, 'square', 'Oversampled')
# Analyze(1.0, 25.0, 1.0, 'square', 'Nyquist')

signal = createSignal('square', timeDomain, signalParts)




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
