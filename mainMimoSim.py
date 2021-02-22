import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import numpy.random as ra


import signalLFM as lfm

# Generic requirements:
# For each Tx Antenna
#   For each Rx Antenna
#       Map the waveform, phase shift[s], and scaling!.

def simulateTxToRx(tx_loc, target_loc, rx_loc, raw_waveform, rcs, fs, fc):
    """
    tx_loc is a 3 array of positions x,y,z
    target_loc is a 3 array of positions x,y,z
    rx_loc is a 3 array of positions x,y,z
    raw_waveform is exactly that
    """

    #d for distance
    d_tx_t = la.norm(tx_loc - target_loc)
    d_t_rx = la.norm(target_loc - rx_loc)
    d_total = d_tx_t + d_t_rx
    wavelength = 3e8/fc

    phase_shift = 2j* np.pi* d_total / wavelength
    n_delay = np.int64(np.round((fs/fc) * (d_total/wavelength)))
    #print(n_delay)
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

    t1_tx_antenna = np.array([0,0,0])
    target_pos = np.array([wl*1.0, 0, 0])
    t1_rx1_antenna = np.array([wl*2.0, 0, 0])
    t1_rx2_antenna = np.array([wl*2.5, 0, 0])
    t1_rx3_antenna = np.array([wl*3.0, 0, 0])

    #Used to show the delays are zero for zero seperation; can't work properly due to div by zero
    rx_div0 = simulateTxToRx(t1_rx1_antenna, t1_rx1_antenna, t1_rx1_antenna, tx, 1, fs, fc)

    #Stock comparison
    rx_direct1 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx1_antenna, tx, 1, fs, fc)
    rx_direct2 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx2_antenna, tx, 1, fs, fc)
    rx_direct3 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx3_antenna, tx, 1, fs, fc)
    rx_via3 = simulateTxToRx(t1_tx_antenna, target_pos, t1_rx3_antenna, tx, 1, fs, fc)

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

simulateTxToRx_test()

def simulateWholeSystem(tx_locs, target_locs, rx_locs, targets, waveforms, noise_power, fs, fc, interf=False):
    """
    tx_locs is a    n_tx * 3        matrix of tx antenna positions
    tx_locs is a    n_targets * 3   matrix of target positions
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

    outputs = np.zeros([n_rx, n_samp])

    if (n_tx != n_waveforms):
        print("WARNING: waveform and TX mismatch")

    for rx in np.arange(n_rx):
        for tx in np.arange(n_tx):
            for target in np.arange(n_targets):
                outputs[rx,:] = simulateTxToRx(tx_locs[tx], target_locs[target],
                    rx_locs[rx], waveforms[tx], targets[target], fs, fc)

    outputs += ra.randn(n_rx, n_samp) * noise_power**(1/2)

    return outputs


def simulateWholeSystem_test():
    fc = 2.4e9
    wl = 3e8/fc
    t_duration = 1e-3
    T = 1e-3
    Tp = 2e-4
    B = 3e6
    fs = 10e6
    A0 = 1
    rcs = np.array([[1]])

    antennas_tx = np.array(([[wl*0.25,0,0],[-wl*0.25,0,0]]))
    target = np.array([[0,10000,0]])
    antennas_rx = np.array([[0,0,0],[wl*0.5,0,0],[-wl*0.5,0,0]])

    pulse_low, ref_low = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 0, 10e6, 1 )
    pulse_high, ref_high = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 5e6, 10e6, 1 )

    waveforms = np.array([pulse_low, pulse_high])
    raw_outputs = simulateWholeSystem(antennas_tx, target, antennas_rx, rcs, waveforms, 0.0, fs, fc)

    plt.plot(10*np.log10(np.abs(nfft.fft(raw_outputs[0]))))
