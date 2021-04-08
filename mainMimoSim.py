import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import numpy.linalg as la
import numpy.random as ra


import signalLFM as lfm


def simulateTxToRx(tx_loc, target_loc, rx_loc, raw_waveform, rcs, fs, fc):
    """
    tx_loc is a 3 array of positions x,y,z
    target_loc is a 3 array of positions x,y,z
    rx_loc is a 3 array of positions x,y,z
    raw_waveform is exactly that
    rcs is the target cross section
    fs is the sampling frequency
    fc is the centre frequency (narrowband assumption)
    """

    #d for distance
    d_tx_t = la.norm(tx_loc - target_loc)
    d_t_rx = la.norm(target_loc - rx_loc)
    d_total = d_tx_t + d_t_rx
    wavelength = 3e8/fc

    phase_shift = 2j* np.pi* d_total / wavelength
    n_delay = np.int64(np.round((fs/fc) * (d_total/wavelength)))
    n_samp = len(raw_waveform)

    #Wavelength is there
    #Divide by 4*pi each area needed.
    #We calculate power loss, need to square it to convert to voltage loss.
    loss_tx_t = (wavelength) / (4 * np.pi * d_tx_t**2)
    loss_rx_t = (wavelength) / (4 * np.pi * d_t_rx**2)
    loss = loss_rx_t * loss_tx_t * (rcs/(4 * np.pi))
    #print("loss = ", loss**(1/2))

    output = np.zeros(len(raw_waveform), dtype = np.complex128)
    output[n_delay:] = raw_waveform[:(n_samp - n_delay)] * loss**(1/2) * np.exp(phase_shift)
    return output;

def simulateTxToRx_test():
    fc = 0.3e9
    fs = 100e6
    wl = 3e8/fc
    rcs = 1

    #Generate a waveform as a chirp
    #This chirp has a duration of 1000us, pulse repitition of 100us,
    #   pulse transmission time of 50us
    # BW and carrier of 1e6, fs of 100e6, and amplitude of 1
    tx, mix = lfm.generateChirpPulsedLFM(0.001, 0.0001, 0.00002, 1e6, 1e6, fs, 1)

    #Tests are just via visual inspection
    #Test 1: Loss and delay between doubly seperated antenns
    #Test 2: Phase difference between far antennas seperated by
    #   fractions of wavelengths
    #Test 3: Making sure that the distances assoc with target work properly
    #Test 4: Making sure that the distance is actually working with correlation

    t1_tx_antenna = np.array([0,0,0])
    target_pos = np.array([wl*1.0, 0, 0])
    target_pos_long = np.array([250, 0, 0])
    t1_rx1_antenna = np.array([wl*2.0, 0, 0])
    t1_rx2_antenna = np.array([wl*2.5, 0, 0])
    t1_rx3_antenna = np.array([wl*3.0, 0, 0])

    #Used to show the delays are zero for zero seperation; can't work properly due to div by zero
    #rx_div0 = simulateTxToRx(t1_rx1_antenna, t1_rx1_antenna, t1_rx1_antenna, tx, 1, fs, fc)

    #Stock comparison
    rx_direct1 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx1_antenna, tx, 1, fs, fc)
    rx_direct2 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx2_antenna, tx, 1, fs, fc)
    rx_direct3 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx3_antenna, tx, 1, fs, fc)
    rx_via3 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx3_antenna, tx, 1, fs, fc)
    rx_long = simulateTxToRx(t1_tx_antenna, target_pos_long, t1_rx1_antenna, tx, 1, fs, fc)

    #Use Correlation to find the results!
    corr = lfm.processLFMPulseCompression(tx,tx,fs,0,3)
    print("Case 1 : Peak power, no phase shift")
    print("|null|, angle(null) = ", np.var(tx), ",", np.angle(corr), "\n")

    corr = lfm.processLFMPulseCompression(tx,rx_direct1,fs,0,3)
    print("Case 2 : Lowered by (4*pi)**-3, minimal phase shift due to 2 lambda seperation")
    print("|rx1|, angle(tx1) = ", np.var(rx_direct1)/np.var(tx), ",", np.angle(corr), "\n")

    corr = lfm.processLFMPulseCompression(tx,rx_direct2,fs,0,3)
    print("Case 3 : Lowered by (1.5)**-2 * (4*pi)**-3, pi phase shift, due to 1 lambda seperation")
    print("|rx2|, angle(tx2) = ", np.var(rx_direct2)/np.var(tx), ",", np.angle(corr), "\n")

    corr = lfm.processLFMPulseCompression(tx,rx_direct3,fs,0,3)
    print("Case 4 : Lowered by (2)**-2 * (4*pi)**-3, no phase shift, due to 1 lambda seperation")
    print("|rx2|, angle(tx2) = ", np.var(rx_direct3)/np.var(tx), ",", np.angle(corr), "\n")

    f_max = 50
    r_max = 1e3
    corr = lfm.processLFMPulseCompression(tx, rx_long, fs, f_max, r_max)
    plt.plot(10*np.log10(np.abs(corr)))
    plt.title("0 doppler shift slice of pulse compression test")
    plt.show()


def simulateWholeSystem(tx_locs, target_locs, rx_locs, targets, waveforms, noise_power, fs, fc, interf=False):
    """
    tx_locs is a    n_tx * 3        matrix of tx antenna positions
    target_locs is  n_targets * 3   matrix of target positions
    rx_locs is a    n_rx * 3        matrix of rx antenna positions
    targets is a    n_targets       vector of target rcs values
    waveforms is a  n_tx * n_samp   matrix of data

    noise power     defines the noise per Rx antenna (scalar)
    interf will be  used later to play with Tx antenna mutual coupling (bool)

    outputs is a    n_rx *n_samp    matrix of received signals

    """
    n_tx = len(tx_locs[:,0])
    n_rx = len(rx_locs[:,0])
    n_waveforms = len(waveforms[:,0])
    n_targets = len(targets)
    n_samp = len(waveforms[0,:])

    outputs = np.zeros([n_rx, n_samp], dtype=np.complex128)

    if (n_tx != n_waveforms):
        print("WARNING: waveform and TX mismatch")

    for rx in np.arange(n_rx):
        for tx in np.arange(n_tx):
            for target in np.arange(n_targets):
                outputs[rx,:] += simulateTxToRx(tx_locs[tx], target_locs[target],
                    rx_locs[rx], waveforms[tx], targets[target], fs, fc)

    outputs += ra.randn(n_rx, n_samp) * noise_power**(1/2)

    return outputs


def simulateWholeSystem_test():
    """
    System aspects I need to verify:

    1) Multiple signals - use bandpass signals, check ffts
    2) Multiple signal extraction - correlate, check
    3) Multiple target returns; seperate RCS values, check.
    4) Noise power; let TX power = 0, measure outputs
    5) Interference; I'll do this one later when I understand it better.
    """
    fc = 2.4e9
    wl = 3e8/fc
    t_duration = 1e-3
    T = 1e-3
    Tp = 2e-4
    B = 5e6
    fs = 20e6
    A0 = 1
    rcs = np.array([1])
    antennas_tx = np.array(([[wl*0.25,0,0],[-wl*0.25,0,0]]))
    target = np.array([[100,0,0]])
    antennas_rx = np.array([[0,0,0],[wl*0.5,0,0],[-wl*0.5,0,0]])

    rcs_multiple = np.array([1,1e-1])
    target_mult = np.array([[100,0,0],[200,0,0]])

    pulse_low, ref_low = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 0, 10e6, 1 )
    pulse_high, ref_high = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 5e6, 10e6, 1 )

    # Case 1
    # Multiple signals, check FFTs.
    check_case_1 = False
    if (check_case_1):
        waveforms_high = np.array([1e-10*pulse_low, pulse_high])
        waveforms_low = np.array([pulse_low, 1e-10*pulse_high])
        waveforms_both = np.array([pulse_low, pulse_high])
        raw_outputs_high = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms_high, 0.0, fs, fc)
        raw_outputs_low = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms_low, 0.0, fs, fc)
        raw_outputs_both = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms_both, 0.0, fs, fc)

        plt.figure()
        plt.plot(10*np.log10(np.abs(nfft.fft(pulse_high))), label="High,TX")
        plt.plot(10*np.log10(np.abs(nfft.fft(pulse_low))), label="Low,TX")
        plt.title("Plot the high and the low in seperate signals to show our ideal")
        plt.legend()

        plt.figure()
        plt.plot(10*np.log10(np.abs(nfft.fft(raw_outputs_high[0]))), label="High")
        plt.title("Run the high frequency signal only through our simulation")
        plt.legend()

        plt.figure()
        plt.plot(10*np.log10(np.abs(nfft.fft(raw_outputs_low[0]))), label="Low")
        plt.title("Run the low frequency signal only through our simulation")
        plt.legend()

        plt.figure()
        plt.plot(10*np.log10(np.abs(nfft.fft(raw_outputs_both[0]))), label="Both")
        plt.title("Run both high and low frequencies signal only through our simulation")
        plt.legend()

    #Case 2
    #Multiple signal extraction; correlate, check. Does phase and presence
    check_case_2 = False;
    if (check_case_2):
        waveforms_0_0 = np.array([pulse_low, pulse_high])
        waveforms_0_pi = np.array([pulse_low, pulse_high*np.exp(1j*np.pi)])
        waveforms_0_blank = np.array([pulse_low, pulse_high*0])

        raw_outputs_0_0 = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms_0_0, 0.0, fs, fc)
        raw_outputs_0_pi = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms_0_pi, 0.0, fs, fc)
        raw_outputs_0_blank = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms_0_blank, 0.0, fs, fc)

        corr_high_0_0 = lfm.processLFMPulseCompression(raw_outputs_0_0[0], pulse_high, fs, 500, 2e3)
        corr_low_0_0 = lfm.processLFMPulseCompression(raw_outputs_0_0[0], pulse_low, fs, 500, 2e3)
        corr_high_0_pi = lfm.processLFMPulseCompression(raw_outputs_0_pi[0], pulse_high, fs, 500, 2e3)
        corr_low_0_pi = lfm.processLFMPulseCompression(raw_outputs_0_pi[0], pulse_low, fs, 500, 2e3)
        corr_high_0_blank = lfm.processLFMPulseCompression(raw_outputs_0_blank[0], pulse_high, fs, 500, 2e3)
        corr_low_0_blank = lfm.processLFMPulseCompression(raw_outputs_0_blank[0], pulse_low, fs, 500, 2e3)

        plt.figure()
        plt.plot(10*np.log10(np.abs(corr_high_0_blank)))
        plt.title("High, 0, blank")

        plt.figure()
        plt.plot(10*np.log10(np.abs(corr_low_0_blank)))
        plt.title("Low, 0, blank")

        plt.figure()
        #plt.imshow(10*np.log10(np.abs(corr_high_0_pi)))
        plt.plot(10*np.log10(np.abs(corr_high_0_0)))
        plt.title("High, 0, 0")

        plt.figure()
        #plt.imshow(10*np.log10(np.abs(corr_low_0_pi)))
        plt.plot(10*np.log10(np.abs(corr_low_0_0)))
        plt.title("Low, 0, 0")

        plt.figure()
        plt.plot(np.angle(corr_high_0_0),label="high 0_0 | Should have same phase at 0")
        plt.plot(np.angle(corr_low_0_0),label="low 0_0 | Should have same phase at 0")
        plt.legend()
        plt.title("Phase comparison")

        plt.figure()
        plt.plot(np.angle(corr_high_0_pi),label="high 0,pi | Should have pi phase diff at 0")
        plt.plot(np.angle(corr_low_0_pi),label="low 0,pi | Should have same phase at 0")
        plt.legend()
        plt.title("Phase comparison, low should be the same as other case")

        plt.show()

    #Case 3
    #Letting there be multiple targets. Also checks that distance works properly!
    check_case_3 = False
    if (check_case_3):
        waveforms = np.array([pulse_low, pulse_high])
        raw_outputs = simulateWholeSystem(antennas_tx, target_mult, antennas_rx, rcs_multiple, waveforms, 0.0, fs, fc)

        plt.figure()
        plt.plot((np.angle(raw_outputs[0])),label = "0")
        plt.plot((np.angle(raw_outputs[1])),label = "1")
        plt.plot((np.angle(raw_outputs[2])),label = "2")
        plt.legend()

        corr_high = lfm.processLFMPulseCompression(pulse_high, raw_outputs[1], fs, 50, 1e3)
        corr_low = lfm.processLFMPulseCompression(pulse_low, raw_outputs[1], fs, 50, 1e3)
        print(np.shape(corr_high))

        plt.figure()
        plt.plot(10*np.log10(np.abs(corr_high)),label = "high")
        plt.plot(10*np.log10(np.abs(corr_low)),label = "low")
        plt.legend()
        plt.title("Testing distances and powers")

        plt.show()

    #Case 4
    #Letting the TX power be zero and measuring the noise
    check_case_4 = False
    if (check_case_4):
        waveforms = np.array([pulse_low*0, pulse_high*0])
        raw_outputs_5 = simulateWholeSystem(antennas_tx, target_mult, antennas_rx, rcs_multiple, waveforms, 5, fs, fc)
        raw_outputs_1em5 = simulateWholeSystem(antennas_tx, target_mult, antennas_rx, rcs_multiple, waveforms, 1e-5, fs, fc)

        print("Noise powers should be ~5:")
        print(np.var(raw_outputs_5[0])," , ",np.var(raw_outputs_5[1])," , ",np.var(raw_outputs_5[2]))
        print("Noise powers should be ~1e-5:")
        print(np.var(raw_outputs_1em5[0])," , ",np.var(raw_outputs_1em5[1])," , ",np.var(raw_outputs_1em5[2]))
