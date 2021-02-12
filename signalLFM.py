import numpy as np
import numpy.random as ra
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


def processLFMFourier(reference, received):
    """
    reference is a   n_samp       vector that is the reference radar signal
    received  is a   n_samp       vector that is the received radar signal
    """

    output_time = reference * received
    output = nfft.fft(output_time)
    return output

def processLFMPulseCompression(reference, received, fs, f_max, r_max):
    """
    Implements pulse compression using the fourier method.
    Remember with delays that n = fs * r/c; s^(-1) * m / ms^(-1)

    reference is a      n_samp   length vector

    """

    #Constants
    c = 3e8
    n_samples_ref = len(reference)
    n_samples_rec = len(received)

    delay_max = np.int64(np.round(fs * r_max/c))

    f_step = fs / (n_samples_ref) #Presumably set to this value because of the limits imposed by the CRB
    print("f_step = ", f_step)
    n_freq_step = np.int64(np.round( f_max/f_step ))

    freq_shifts = np.arange(-n_freq_step, n_freq_step + 1, 1)

    output = np.zeros((delay_max, 2 * n_freq_step + 1), dtype = np.complex128)

    reference_f = nfft.fft(reference)
    received_f = nfft.fft(received)

    for shift in freq_shifts:
        #Multiply (in frequency domain, time domain convolution)
        output_f = np.conj(np.roll(reference_f,shift)) * received_f
        #Convert back to time domain and add to the output vector:
        output[:,shift + n_freq_step] = nfft.ifft(output_f)[:delay_max]

    return output


def processLFMPulseCompression_test():
    """
    Exists to test the pulse compression. Specifically:
    If correlation produces a peak
    If it doesn't product a peak for uncorr signals
    If the delay works
    If the phase extraction works
    If the frequency shift works.

    No automatic testing, based purely on visual inspection.
    """

    awgn = ra.randn(100000) + 1j*ra.randn(100000)
    awgn_roll = np.roll(awgn, 12) #Delay of 12 corresponds to 30*12 = 360m?
    awgn_phase = np.exp(1j*1.345)*awgn
    awgn_uncorr = ra.randn(100000) + 1j*ra.randn(100000)
    f_shift = 40
    f_s = 2e6   #n_samp = 100e3, fs=2e6 gives a doppler
                #   step of 20Hz in the fourier domain
    awgn_doppler = awgn * np.exp(2j*np.pi*(f_shift/f_s)*np.arange(100000))

    plt.plot(10*np.log10(np.abs(nfft.fft(awgn))))
    plt.plot(10*np.log10(np.abs(nfft.fft(awgn_doppler))))
    plt.show()

    output_same = processLFMPulseCompression(awgn, awgn, f_s, 500, 1e3)
    output_roll = processLFMPulseCompression(awgn, awgn_roll, f_s, 500, 1e3)
    output_uncorr = processLFMPulseCompression(awgn, awgn_uncorr, f_s, 500, 1e3)
    output_phase = processLFMPulseCompression(awgn, awgn_phase, f_s, 500, 1e3)
    output_doppler = processLFMPulseCompression(awgn, awgn_doppler, f_s, 500, 1e3)

    plt.figure()
    plt.imshow(10*np.log10(np.abs(output_same)))
    plt.title("Same")
    plt.figure()
    plt.imshow(10*np.log10(np.abs(output_roll)))
    plt.title("Delay")
    plt.figure()
    plt.imshow(10*np.log10(np.abs(output_uncorr)))
    plt.title("Uncorr")
    plt.figure()
    plt.imshow(10*np.log10(np.abs(output_doppler)))
    plt.title("Dopp")
    plt.figure()
    plt.imshow(np.angle(output_phase))
    plt.title("Phase")

    plt.show()
