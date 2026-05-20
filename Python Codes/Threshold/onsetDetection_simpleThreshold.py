import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from emg_utils import compute_rms, compute_tkeo, detect_phases, compute_threshold

# Parameter
ANALOG_RATE  = 2000
WINDOW_MS    = 50

window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE) #rechnet Fenstergröße von ms in Samples um


def onsetDetection_simpleThresholdRMS(file_path, threshold_std):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE  #Zeitachse für Rohsignal
        envelope           = compute_rms(signal, window_samples)
        threshold          = compute_threshold(envelope, threshold_std)
        all_phases[muscle] = detect_phases(envelope, threshold)

    plot_onset(emg_data, all_phases, t_signal, "Simple Threshold RMS")

def onsetDetection_simpleThresholdTKEO(file_path, threshold_std):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE  #Zeitachse für Rohsignal
        envelope           = compute_tkeo(signal, window_samples)
        threshold          = compute_threshold(envelope, threshold_std)
        all_phases[muscle] = detect_phases(envelope, threshold)

    plot_onset(emg_data, all_phases, t_signal, "Simple Threshold TKEO")


# Aufruf
FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
onsetDetection_simpleThresholdRMS(FILE_PATH, threshold_std=5)
#onsetDetection_simpleThresholdTKEO(FILE_PATH, threshold_std=5)
