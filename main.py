import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import numpy.linalg as la
import numpy.random as ra


import signalLFM as lfm
import mainMimoProcessing as pro
import mainMimoSim as sim

def mainRawSeeringVectors():
    """
    Places a target at some distance, and attempts to use naive steering
        vectors to extract information, ie is non adaptive. The purpose of this
        this simulation is to demonstate the increases in [virtual] aperture size
        given by MIMO.

    Let the RX array be spaced appropriately to avoid grating lobes
    Let the TX array be spaced appropriate to basically duplicate the array
        n_rx times
    Let the system be configured broadside to maximise the desired effect.
        Thus, array spans the +-x, and the target is at the +-y.

    """

    fc = 2.4e9
    wl = 3e8/fc
    hl = wl/2 #Half wavelength. Short definition to minimise code line length
    t_duration = 1e-3
    T = 1e-3
    Tp = 2e-4
    B = 2.5e6
    fs = 10e6
    A0 = 1
    rcs = np.array([[1]])
    target = np.array([[0,100,0]])

    pulse_0, ref_0 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 0, fs, A0 )
    pulse_1, ref_1 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 2.5e6, fs, A0 )
    pulse_2, ref_2 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 5e6, fs, A0 )
    pulse_3, ref_3 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 7.5e6, fs, A0 )

    rx_five = np.array([[2*hl,0,0],[hl,0,0],[0,0,0],[-hl,0,0],[-2*hl,0,0]])
    tx_solo = np.array([[0,0,0]])
    tx_two  = np.array([[2.5*hl,0,0],[-2.5*hl,0,0]])
    tx_four = np.array([[5*hl,0,0],[2.5*hl,0,0],[-2.5*hl,0,0],[-5*hl,0,0]])

    waveform_solo = np.array([pulse_0])
    waveform_two = np.array([pulse_0, pulse_1])
    waveform_four = np.array([pulse_0, pulse_1, pulse_2, pulse_3])

    outputs_solo = sim.simulateWholeSystem(tx_solo, target, rx_five, rcs, waveform_solo, 0.0, fs, fc)
    outputs_two = sim.simulateWholeSystem(tx_two, target, rx_five, rcs, waveform_two, 0.0, fs, fc)
    outputs_four = sim.simulateWholeSystem(tx_four, target, rx_five, rcs, waveform_four, 0.0, fs, fc)

    y_solo, peak_solo = pro.extractPeakRangeDoppler(outputs_solo, waveform_solo, fs, 10, 400)
    y_two, peak_two = pro.extractPeakRangeDoppler(outputs_two, waveform_two, fs, 10, 400)
    y_four, peak_four = pro.extractPeakRangeDoppler(outputs_four, waveform_four, fs, 10, 400)

    angles = np.deg2rad(np.arange(0,180))

    results_solo = np.zeros(len(angles))
    results_two = np.zeros(len(angles))
    results_four = np.zeros(len(angles))

    angle_show = np.zeros( (5*4,len(angles)), dtype = np.complex128)

    for angle in range(len(angles)):
        #Generate steering vectors
        s_solo = pro.generateMimoSteeringVector(angles[angle], 0, tx_solo, rx_five, fc)
        s_two = pro.generateMimoSteeringVector(angles[angle], 0, tx_two, rx_five, fc)
        s_four = pro.generateMimoSteeringVector(angles[angle], 0, tx_four, rx_five, fc)

        #Debug: display s_four across time
        angle_show[:,angle] = s_four

        #Mimo beamforming
        results_solo[angle] = (np.abs(np.inner(s_solo,y_solo)))
        results_two[angle] = (np.abs(np.inner(s_two,y_two)))
        results_four[angle] = (np.abs(np.inner(s_four,y_four)))

    plt.figure()
    plt.plot(np.rad2deg(angles), 20*np.log10(results_solo/max(results_solo)), label = "solo")
    plt.plot(np.rad2deg(angles), 20*np.log10(results_two/max(results_two)), label = "two")
    plt.plot(np.rad2deg(angles), 20*np.log10(results_four/max(results_four)), label = "four")
    plt.legend()
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Power")

    plt.figure()
    plt.imshow(np.angle(angle_show))
    plt.show()

def mainSuperdirectiveSNR():
    """
    Places a target some distance from the antenna, and observes how well
      variations in the steering vector angle will affect reception.
    The goal is to see how variations in the stabilising parameter affect
      the power of the target
    """

    fc = 2.4e9
    wl = 3e8/fc
    hl = wl/2 #Half wavelength. Short definition to minimise code line length
    t_duration = 1e-3
    T = 1e-3
    Tp = 2e-4
    B = 2.5e6
    fs = 10e6
    A0 = 1
    rcs = np.array([[1]])
    target = np.array([[0,100,0]])
    phi = 0

    pulse_0, ref_0 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 0, fs, A0 )
    pulse_1, ref_1 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 2.5e6, fs, A0 )
    pulse_2, ref_2 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 5e6, fs, A0 )
    pulse_3, ref_3 = lfm.generateChirpPulsedLFM(t_duration, T, Tp, B, 7.5e6, fs, A0 )

    rx_five = np.array([[2*hl,0,0],[1*hl,0,0],[0,0,0],[-1*hl,0,0],[-2*hl,0,0]])
    tx_four = np.array([[4.5*hl,0,0],[2.5*hl,0,0],[-2.5*hl,0,0],[-4.5*hl,0,0]])

    rx_tx_virtual = np.kron(tx_four[:,0], rx_five[:,0])

    waveform_four = np.array([pulse_0, pulse_1, pulse_2, pulse_3])
    outputs_four = sim.simulateWholeSystem(tx_four, target, rx_five, rcs, waveform_four, 0.0, fs, fc)
    y_four, peak_four = pro.extractPeakRangeDoppler(outputs_four, waveform_four, fs, 10, 400)
    angles = np.deg2rad(np.arange(0,180,0.25))

    results_sv = np.zeros(len(angles))
    results_10 = np.zeros(len(angles))
    results_15 = np.zeros(len(angles))
    results_20 = np.zeros(len(angles))

    R = pro.generateCovarianceIsotropic(tx_four, rx_five, fc)

    for angle in range(len(angles)):
        s_sv = pro.generateMimoSteeringVector(angles[angle], phi, tx_four, rx_five, fc)
        s_10 = pro.generateSuperdirective(R, 10, angles[angle], phi, tx_four, rx_five, fc)
        s_15 = pro.generateSuperdirective(R, 16, angles[angle], phi, tx_four, rx_five, fc)
        s_20 = pro.generateSuperdirective(R, 20, angles[angle], phi, tx_four, rx_five, fc)

        results_sv[angle] = (np.abs(np.inner(s_sv,y_four)))
        results_10[angle]  = (np.abs(np.inner(s_10,y_four)))
        results_15[angle] = (np.abs(np.inner(s_15,y_four)))
        results_20[angle] = (np.abs(np.inner(s_20,y_four)))

    plt.figure()
    plt.plot(np.rad2deg(angles), 20*np.log10(results_10/max(results_10)), label = "10")
    plt.plot(np.rad2deg(angles), 20*np.log10(results_sv/max(results_sv)), label = "sv")
    plt.legend()
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Power")

    plt.figure()
    plt.plot(np.rad2deg(angles), 20*np.log10(results_10/max(results_10)), label = "10")
    plt.plot(np.rad2deg(angles), 20*np.log10(results_20/max(results_20)), label = "20", color="green")
    plt.legend()
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Power")
    plt.show()

mainSuperdirectiveSNR()
