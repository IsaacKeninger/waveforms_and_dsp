import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft, fftfreq
import sys

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
    
    if (waveform.lower() == "sin" ):
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
def plotGraph(freqBins, mags, time_domain, signal, sampleRateType):

    fig, (plt1, plt2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Plot Time Domain
    plt1.plot(time_domain, signal)
    plt1.set_title(f"Time Domain: {sampleRateType}")
    plt1.set_xlabel("Duration (secs)")
    plt1.set_ylabel("Amplitude")

    # Plot frequency domain
    plt2.plot(freqBins, mags)
    plt.xlim(0, max(freqBins))
    plt2.set_xlabel("Frequencies (Hz)")
    plt2.set_ylabel("Amplitude (magnitude)")
    plt2.set_title(f"Frequency Domain: {sampleRateType}")

    fig.tight_layout()

def Analyze(amplitude, frequency, duration, waveform, sampleRateType):

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

    # Convert Signal to frequency domain and track magnitudes at each freuqnecy
    signal_magnitudes = frequencyDomain_transfer(signal)

    # Make section domain for certain frequenceis
    frequnecy_bins_sineUndersample = make_frequency_bins(signal, sampleRate)

    # PLOT GRAPH
    plotGraph(frequnecy_bins_sineUndersample, signal_magnitudes, timeDomain, signal, sampleRateType)

# Sin waves
Analyze(1.0, 25.0, 1.0, 'sin', 'Undersampled')
Analyze(1.0, 25.0, 1.0, 'sin', 'Proper')
Analyze(1.0, 25.0, 1.0, 'sin', 'Oversampled')
Analyze(1.0, 25.0, 1.0, 'sin', 'Nyquist')

# # Saw Waves
# Analyze(1.0, 25.0, 1.0, 'saw', 'Undersampled')
# Analyze(1.0, 25.0, 1.0, 'saw', 'Proper')
# Analyze(1.0, 25.0, 1.0, 'saw', 'Oversampled')
# Analyze(1.0, 25.0, 1.0, 'saw', 'Nyquist')

# # Square Waves
# Analyze(1.0, 25.0, 1.0, 'square', 'Undersampled')
# Analyze(1.0, 25.0, 1.0, 'square', 'Proper')
# Analyze(1.0, 25.0, 1.0, 'square', 'Oversampled')
# Analyze(1.0, 25.0, 1.0, 'square', 'Nyquist')

plt.show()

# Resources: 
#   https://matplotlib.org/stable/index.html
#   https://docs.scipy.org/doc/scipy/
#   https://numpy.org/doc/2.3/user/index.html#user
#   https://stackoverflow.com/questions/543309/programmatically-stop-execution-of-python-script
#   https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
#   https://www.dspguide.com/ch3/2.htm
#   https://pysdr.org/content/sampling.html
#   https://dsp.stackexchange.com/questions/26927/what-is-a-frequency-bin
#   https://dsp.stackexchange.com/questions/26927/what-is-a-frequency-bin 



