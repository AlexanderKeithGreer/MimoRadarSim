import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import numpy.linalg as la
import numpy.random as ra

import signalLFM as lfm

def generateSteeringVector(theta, phi, array, fc):
    """
    INPUTS:
    theta is a      scalar          parameter to define azimuth
    phi   is a      scalar          parameter to define altitude
    array is a      n_elem * 3      matrix of antenna locations

    OUTPUTS:
    s is a          n_elem * 1      vector of phase shifts, (a steering vector)
    """
    n_elem = np.shape(array)[0] #The number of rows in the array matrix
    c = 3e8

    #s is the steering vector
    #k is the [unit] look direction vector
    s = np.zeros((n_elem,1),dtype=np.complex128)
    k = np.array([np.cos(theta)*np.cos(phi),np.sin(theta)*np.cos(phi),np.sin(phi)])

    for elem in range(n_elem):
        s[elem] = np.exp(-2j*np.pi*(c/fc)*np.dot(array[elem,:],k))

    return s

def generateMimoSteeringVector(theta, phi, array_tx, array_rx, fc):
    """
    Assumes (mostly) co-located arrays

    INPUTS:
    theta is a      scalar          parameter to define azimuth
    phi is a        scalar          parameter to define altitude
    array_tx is a   n_elem_tx * 3   matrix of tx antenna locations
    array_rx is a   n_elem_rx * 3   matrix of rx antenna locations

    OUTPUTS:
    s is a          n_elem_tx * n_elem_rx   mimo steering vector
    """

    s_tx = generateSteeringVector(theta, phi, array_tx, fc)
    s_rx = generateSteeringVector(theta, phi, array_rx, fc)
    # Form the mimo steering vector. Convertion is s_tx kron s_rx
    s = np.kron(s_tx, s_rx).flatten()
    s = np.conj(s)

    return s

def extractPeakRangeDoppler(raw_waveforms, ref_waveforms, fs, f_max, r_max):
    """
    Performs pulse compression, and then return a bunch of complex numbers
        from the peak of our correlation. This is only going to work for
        one target, which must be distinguishable from noise.

    INPUTS:
    raw_waveforms is an n_rx * n_samp   matrix of received signals
    ref_waveforms is a  n_tx * n_samp   matrix of transmitted signals
    fs is a             scalar          parameter, for sampling frequency
    f_max is a          scalar          denoting the peak doppler shift
    r_max is a          scalar          denoting the peak range

    OUTPUTS:
    y is a              n_tx * n_rx     vector with the target amplitudes and phases
    """

    n_tx = np.shape(ref_waveforms)[0]
    n_rx = np.shape(raw_waveforms)[0]
    y = np.zeros(n_tx * n_rx, dtype = np.complex128)

    for tx in range(n_tx):
        for rx in range(n_rx):
            rd_matrix = lfm.processLFMPulseCompression(ref_waveforms[tx], raw_waveforms[rx], fs, f_max, r_max)
            peak = np.argmax(np.abs(rd_matrix))
            y[rx + tx * n_rx] = rd_matrix.flatten()[peak]

    return y, peak
