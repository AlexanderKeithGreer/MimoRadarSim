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
        s[elem] = np.exp(2j*np.pi*np.dot((fc/c)*array[elem,:],k))

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


def generateMimoSteeringVector_test():
    """
    Visual verification of the generateMimoSteeringVector
     function. Intended to ensure that those vectors
    """
    fc = 2.4e9 #2.4GHz
    c = 3e8 #I hope this hasn't changed much
    hl = fc*c /2
    array_tx = np.array([[hl*0.25,0,0],[hl*0.25,0,0]])
    array_tx = np.array([[hl*0.25,0,0],[hl*0.25,0,0]])

def generateVirtualArray(array_tx, array_rx):
    """
    INPUTS:
    array_tx is a      n_tx * 3      matrix of antenna locations
    array_rx is a      n_rx * 3      matrix of antenna locations

    OUTPUTS:
    s is a             n_tx*n_rx    matrix of virtual antenna locations
    """
    n_tx = np.shape(array_tx)[0]
    n_rx = np.shape(array_rx)[0]
    array_vl = np.zeros([n_tx*n_rx,3])

    for tx in range(n_tx):
        for rx in range(n_rx):
            array_vl[tx*n_rx+rx-1,:] = array_tx[tx,:] + array_rx[rx,:]
    return array_vl

def generateVirtualArray_test():
    """ """
    fc = 2.4e9
    wl = 3e8/fc
    hl = wl/2 #Half wavelength. Short definition to minimise code line length

    rx_five = np.array([[3.0*hl,0,0],[1.5*hl,0,0],[0,0,0],[-1.5*hl,0,0],[-3.0*hl,0,0]])
    rx_one = np.array([[0,0,0],[2*hl,0,0]])
    tx_four = np.array([[0.75*hl,0,0],[0.25*hl,0,0],[-0.25*hl,0,0],[-0.75*hl,0,0]])
    targetTheta = 90*np.pi/180
    target = np.array([[1000*np.cos(targetTheta),1000*np.sin(targetTheta),0]])


    rx_tx_virtual = generateVirtualArray(tx_four, rx_one)

    plt.figure()
    plt.scatter(target[:,0], target[:,1])
    plt.scatter(tx_four[:,0], tx_four[:,1])
    plt.scatter(rx_one[:,0], rx_one[:,1])
    plt.figure()
    plt.scatter(target[:,0], target[:,1])
    plt.scatter(rx_tx_virtual[:,0], rx_tx_virtual[:,1])
    plt.show()


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

def sincComplex(inputNo):
    """
    Equivalent to sinc function in accoustic SDB, but works with complex numbers
      to give elements in a coherence matrix. Complex part is the Hilbert
      transform of a sinc,  (1-cos(x))/x.
    Do remember that numpy version of sinc is sin(pi*x)/(pi*x)

    INPUTS:
      inputNo is a    scalar      real number
    OUTPUTS:
      output is a   scalar      real number
    """
    if (inputNo != 0):
        output = np.sinc(inputNo) + 1j*(1-np.cos(np.pi*inputNo))/(np.pi*inputNo)
    else:
        output = 1
    return output

def sincComplex_test():
    """
    Relatively quick check to ensure that the function defined works as expected
    """
    x = np.arange(-10,10,0.01)
    y = np.zeros(len(x), dtype=np.complex128)
    for ii in range(len(x)):
        y[ii] = sincComplex(x[ii])

    plt.plot(x,np.angle(y), label="arg")
    plt.plot(x,10*np.log10(np.real(y)+1e-10),label="R")
    plt.plot(x,10*np.log10(np.imag(y)+1e-10),label="I")
    plt.plot(x,10*np.log10(np.abs(y)+1e-10),label="Mag")
    plt.legend()
    plt.show()

def generateCovarianceIso(array, fc):
    """
    Isotropic noise model is based on Ch 2 of Brandstein's Microphone Array
      Signal processing.

    INPUTS:
    array_tx is a   n_elem x 3      matrix of tx antenna locations
    fc is a         scalar          parameter for the carrier frequency

    OUTPUTS:
    R is a          n_elem x n_elem   matrix (covariance matrix)
    """
    n_elem = np.shape(array)[0]       #number of virtual elements
    c = 3e8

    R = np.zeros([n_elem, n_elem],dtype=np.complex128)
    l = np.zeros([n_elem, n_elem],dtype=np.complex128)

    #This is a block matrix -- it can -probably- be kroned together
    for N in range(n_elem):
        for M in range(n_elem):
            l[M,N] = (fc/c)*la.norm(array[M,:] - array[N,:])
            R[M,N] = np.sinc((fc/c)*la.norm(array[M,:] - array[N,:]))

    return R



def generateCovarianceIsoMimo(array_tx, array_rx, fc):
    """
    Assumes (mostly co-located arrays). TX is the major dimension.
    Isotropic noise model is based on Ch 2 of Brandstein's Microphone Array
      Signal processing.
    It is based on generating the covariance matrices associated with
      the TX and RX and kron'ing them together

    INPUTS:
    array_tx is a   n_elem_tx x 3   matrix of tx antenna locations
    array_rx is a   n_elem_rx x 3   matrix of rx antenna locations
    fc is a         scalar          parameter for the carrier frequency

    OUTPUTS:
    R is a          (n_elem_tx*n_elem_rx)   matrix (covariance matrix)
                    x (n_elem_tx*n_elem_rx)
    """
    R_tx = generateCovarianceIso(array_tx, fc)
    R_rx = generateCovarianceIso(array_rx, fc)

    R = np.kron(R_tx, R_rx)
    return R

def generateSuperdirectiveVirtual(SNR, theta, phi, array_tx, array_rx, fc):
    """
    Assumes (mostly) co-located arrays. TX is the major dimension.
    Solves the MVDR using the covariance matrix assoc with an
    isotropic noise field.

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
    array_vl = generateVirtualArray(array_tx, array_rx)

    R = generateCovarianceIso(array_vl, fc)
    R += np.identity(n_elem) * (1/10**(SNR/10))
    R_inv = la.inv(R)
    Ra = np.matmul(R_inv, np.array([a]).T) #Not a Hermitian! Just need to covert to 2d
    s = Ra/np.matmul(a.T.conj(), Ra)
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
    R_two   = generateCovarianceIsoMimo(tx_two,  rx_five, fc)
    s_two   = generateSuperdirective(R_two, 20, theta, phi, tx_two, rx_five, fc)
    R_four  = generateCovarianceIsoMimo(tx_four, rx_five, fc)
    s_four  = generateSuperdirective(R_four, 20, theta, phi, tx_four, rx_five, fc)
    plt.figure()
    plt.imshow(np.angle(R_four))
    plt.figure()
    plt.imshow(np.abs(R_four))
    plt.show()
    print(s_two)
    print(s_four)

def generateSuperdirectiveKron(SNR, theta, phi, array_tx, array_rx, fc):
    """
    Assumes (mostly) co-located arrays. TX is the major dimension.
    Solves the MVDR using the covariance matrix assoc with an
    isotropic noise field.

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
    n_tx = np.shape(array_tx)[0]
    n_rx = np.shape(array_rx)[0]
    s_tx = generateSuperdirective(0, SNR, theta, phi, array_tx, fc)
    s_rx = generateSuperdirective(0, SNR, theta, phi, array_rx, fc)
    s = np.kron(s_tx, s_rx).flatten()
    return s

def generateSuperdirective(R, SNR, theta, phi, array, fc):
    """
    Solves the MVDR using the covariance matrix assoc with an
    isotropic noise field.

    INPUTS:
    array is a      n_elem x    3           matrix of tx antenna locations
    R is a          (n_elem x n_elem)       matrix (covariance matrix)
    fc is a         scalar                  carrier frequency
    theta is a      scalar                  parameter to define azimuth
    phi is a        scalar                  parameter to define altitude
    SNR is a        scalar                  parameter used for robustness (dB)


    OUTPUTS:
    s is a          n_elem                  mimo steering vector
    """
    n_elem = len(array)
    a = generateSteeringVector(theta, phi, array, fc)
    R = generateCovarianceIso(array, fc)
    R += np.identity(n_elem) * (1/10**(SNR/10))
    R_inv = la.inv(R)
    Ra = np.matmul(R_inv, a) #Not a Hermitian! Just need to covert to 2d
    s = Ra/np.matmul(a.T.conj(), Ra)
    s = s.flatten()
    return s

def generateSuperdirectiveKron_test():
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
    R_two   = generateCovarianceIsoMimo(tx_two,  rx_five, fc)
    s_two   = generateSuperdirective(R_two, 20, theta, phi, tx_two, rx_five, fc)
    R_four  = generateCovarianceIsoMimo(tx_four, rx_five, fc)
    s_four  = generateSuperdirective(R_four, 20, theta, phi, tx_four, rx_five, fc)
    plt.figure()
    plt.imshow(np.angle(R_four))
    plt.figure()
    plt.imshow(np.abs(R_four))
    plt.show()
    print(s_two)
    print(s_four)
