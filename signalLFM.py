import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nfft

def generateChirpPulsedLFM(t ,T, Tp, B, fc, fs, A0, verbose = False):
    """
    T = Pulse train period (s)
    Tp = Pulse duration time (s)
    B = bandwidth
    fc = carrier frequency (probably should leave this at zero m8)
    fs = sampling frequency
    """

    if (fs < B):
        print("WARNING: sampling frequency is less than bandwidth")
    if (T < Tp):
        print("WARNING: Pulse repitition time less than pulse duration")

    alpha = B / Tp
    n_samp = int(np.round(t*fs))
    n_pulses = int(np.ceil(t/T))
    n_samp_pulse_on = int(np.floor(Tp*fs))
    n_samp_ref_on = int(np.floor(T*fs))

    if (verbose):
        print("alpha = ", alpha)
        print("n_samp = ", n_samp)
        print("n_pulses = ", n_samp)


    #Generate a copy of the pulses (+reference)
    time = np.arange(0,T,1/fs, dtype = np.complex128)
    chirp = A0*np.exp(2j*np.pi*fc*time + 1j*np.pi*alpha*(time**2) )

    #Generate data vectors
    pulsedLFM = np.zeros(n_samp, dtype=np.complex128)
    referenceLFM = np.zeros(n_samp, dtype=np.complex128)

    #Copy it to the relevant parts of the time
    for pulse in range(n_pulses):
        #Find the start of the current pulse
        pulse_start = int(pulse * T * fs)
        #find the end of the current pulse
        pulse_end = int(pulse_start + (Tp * fs)) - 1
        reference_end = int(pulse_start + (T * fs)) - 1
        #Check we have enough room and resize if not
        if (n_samp <= pulse_end):
            pulse_end = (n_samp - 1)
        if (n_samp <= reference_end):
            pulse_end = (reference_end - 1)

        print(pulse_start)
        print(pulse_end)
        print(pulsedLFM[pulse_start:pulse_end])
        print(chirp[0:n_samp_pulse_on])

        pulsedLFM[pulse_start:pulse_end] = chirp[0:(n_samp_pulse_on-1)]
        referenceLFM[pulse_start:reference_end] = chirp[0:(n_samp_ref_on-1)]

    return pulsedLFM, referenceLFM

def generateChirpPulsedLFM_test():
    tx, mix = generateChirpPulsedLFM(0.001, 0.0001, 0.00001, 1e6, 1e6, 200e6, 1)
    plt.figure()
    plt.plot(mix,label="mix")
    plt.plot(tx,label="tx")
    plt.legend()

    plt.figure()

    plt.plot(10*np.log10(np.abs(nfft.fft(mix)+1e-20)), label="mix")
    plt.plot(10*np.log10(np.abs(nfft.fft(tx)+1e-20)), label="tx")
    plt.show()
