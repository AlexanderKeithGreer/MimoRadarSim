import numpy as np
import numpy.random as ra

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
    d_tx_t = norm(tx_loc - target_loc)
    d_t_rx = norm(target_loc - rx_loc)
    d_total = d_tx_t + d_t_rx
    wavelength = 3e8/fc

    phase_shift = 2* np.pi* d_total / wavelength
    n_delay = (fs/fc) * (d_total/wavelength)
    loss_tx_t = 4**2 * np.pi**2 * d_tx_t**2 / (wavelength**2)
    loss_rx_t = 4**2 * np.pi**2 * d_t_rx**2 / (wavelength**2)
    loss = loss_rx_t * loss_tx_t * rcs

    output = np.zeros(len(raw_waveform))
    output[n_delay:] raw_waveform * loss * np.exp(phase_shift)[:n_delay]
    return output;

def simulate_whole_system(tx_locs, rx_locs, targets, waveforms, noise_power, interf=False):
    """
    tx_locs is a    n_tx * 3        matrix of tx antenna positions
    rx_locs is a    n_rx * 3        matrix of rx antenna positions
    targets is a    n_targets       vector of target rcs values
    waveforms is a  n_tx * n_samp   matrix of data
    noise power defines the noise per Rx antenna (scalar)
    interf will be used later to play with Tx antenna mutual coupling (bool)
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
                outputs[rx,:] = simulateTxToRx(tx_locs[tx], rx_locs[rx], targets[target], waveforms[target])

    outputs += ra.randn(n_rx, n_samp) * noise_power**(1/2)
