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

    phase_shift = 2* np.pi* d_total / wavelength
    n_delay = (fs/fc) * (d_total/wavelength)
    loss_tx_t = 4**2 * np.pi**2 * d_tx_t**2 / (wavelength**2)
    loss_rx_t = 4**2 * np.pi**2 * d_t_rx**2 / (wavelength**2)
    loss = loss_rx_t * loss_tx_t * rcs

    output = np.zeros(len(raw_waveform))
    output[n_delay:] = raw_waveform * loss * np.exp(phase_shift)[:n_delay]
    return output;

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

simulateWholeSystem_test()
