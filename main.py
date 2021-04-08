import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import numpy.linalg as la
import numpy.random as ra


import signalLFM as lfm
import mainMimoSim as sim

def main_raw_steering_vectors():
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

    ouputs_solo = simulateWholeSystem(tx_solo, target, rx_five, rcs, waveform_solo, 0.0, fs, fc)
    ouputs_two = simulateWholeSystem(tx_two, target, rx_five, rcs, waveform_two, 0.0, fs, fc)
    ouputs_four = simulateWholeSystem(tx_four, target, rx_five, rcs, waveform_four, 0.0, fs, fc)
