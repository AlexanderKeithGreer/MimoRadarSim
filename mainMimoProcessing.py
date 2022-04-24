import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import numpy.linalg as la
import numpy.random as ra

import signalLFM as lfm

#Note that most of these functions have no theFunction_test function.
#This is because they are tested by the simulation itself

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
    s = np.zeros((n_elem,1), dtype=np.complex128)
    k = np.array([np.cos(theta)*np.cos(phi),np.sin(theta)*np.cos(phi),np.sin(phi)])
    for elem in range(n_elem):
        s[elem] = np.exp(-2j*np.pi*np.dot((fc/c)*array[elem,:],k))

    return s

def generateMimoSteeringVector(theta, phi, array_tx, array_rx, fc):
    """
    Assumes (mostly) co-located arrays. TX is the major dimension.

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

def generateCovarianceIsotropic(array_tx, array_rx, fc):
    """
    Assumes (mostly co-located arrays). TX is the major dimension.
    Isotropic noise model is based on Ch 2 of Brandstein's Microphone Array
      Signal processing.

    INPUTS:
    array_tx is a   n_elem_tx x 3   matrix of tx antenna locations
    array_rx is a   n_elem_rx x 3   matrix of rx antenna locations
    fc is a         scalar          parameter for the carrier frequency

    OUTPUTS:
    R is a          (n_elem_tx*n_elem_rx)   matrix (covariance matrix)
                    x (n_elem_tx*n_elem_rx)
    """
    n_tx = np.shape(array_tx)[0]
    n_rx = np.shape(array_rx)[0]
    n_elem = n_tx * n_rx         #number of virtual elements
    c = 3e8

    R_tx = np.zeros([n_tx, n_tx])
    R_rx = np.zeros([n_rx, n_rx])
    #This is a block matrix -- it can -probably- be kroned together
    for N in range(n_tx):
        for M in range(n_tx):
            R_tx[M,N] = np.sinc(2*np.pi*np.dot((fc/c)*array_tx[M,:], array_tx[N,:]))
    for N in range(n_rx):
        for M in range(n_rx):
            R_rx[M,N] = np.sinc(2*np.pi*np.dot((fc/c)*array_rx[M,:], array_rx[N,:]))
    R = np.kron(R_tx, R_rx)
    return R

def generateSuperdirective(R, SNR, theta, phi, array_tx, array_rx, fc):
    """
    Assumes (mostly) co-located arrays. TX is the major dimension.
    Solves the MVDR using isotropic

    INPUTS:
    array_tx is a   n_elem_tx x 3           matrix of tx antenna locations
    array_rx is a   n_elem_rx x 3           matrix of rx antenna locations
    R is a          (n_elem_tx*n_elem_rx)   matrix (covariance matrix)
                    x (n_elem_tx*n_elem_rx)
    fc is a         scalar                  carrier frequency
    theta is a      scalar                  parameter to define azimuth
    phi is a        scalar                  parameter to define altitude
    SNR is a        scalar                  parameter used for robustness (dB)


    OUTPUTS:
    s is a          n_elem_tx * n_elem_rx   mimo steering vector
    """
    n_tx = len(array_tx)
    n_rx = len(array_rx)
    n_elem = len(array_tx) * len(array_rx)
    a = generateMimoSteeringVector(theta, phi, array_tx, array_rx, fc)
    R = generateCovarianceIsotropic(array_tx, array_rx, fc)
    R += np.identity(n_elem) * (1/10**(SNR/10))
    R_inv = la.inv(R)
    Ra = np.matmul(R_inv, np.array([a]).T) #Not a Hermitian! Just need to covert to 2d
    s = Ra/np.matmul(Ra.T.conj(), a)
    s = s.flatten()
    return s

def generateSuperdirective_test():
    """
    Quickly check the geometry and coefficient values
    """
    theta = 35
    phi = 0
    fc = 2.4e9
    wl = 3e8/fc
    hl = wl/2 #Half wavelength. Short definition to minimise code line length
    rx_five = np.array([[2*hl,0,0],[hl,0,0],[0,0,0],[-hl,0,0],[-2*hl,0,0]])
    tx_two  = np.array([[2.5*hl,0,0],[-2.5*hl,0,0]])
    tx_four = np.array([[5*hl,0,0],[2.5*hl,0,0],[-2.5*hl,0,0],[-5*hl,0,0]])
    R_two   = generateCovarianceIsotropic(tx_two,  rx_five, fc)
    s_two   = generateSuperdirective(R_two, tx_two,  rx_five, fc, theta, phi, 20)
    R_four  = generateCovarianceIsotropic(tx_four, rx_five, fc)
    s_four  = generateSuperdirective(R_two, tx_four, rx_five, fc, theta, phi, 20)
    print(s_two)
    print(s_four)
