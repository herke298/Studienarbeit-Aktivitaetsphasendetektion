import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from plot_doubleThreshold import plot_doubleThreshold_slider
from emg_utils import compute_rms, compute_tkeo, compute_doubleThreshold, detect_doublePhases

# Parameter
ANALOG_RATE  = 2000
WINDOW_MS    = 50
BASELINE_S   = 1.0

window_samples   = int(WINDOW_MS / 1000 * ANALOG_RATE) #rechnet Fenstergröße von ms in Samples um
baseline_samples = int(BASELINE_S * ANALOG_RATE)


def onsetDetection_doubleThresholdRMS(file_path, t1_std, t2_std):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_rms(signal, window_samples)
        t1, t2             = compute_doubleThreshold(envelope, t1_std, t2_std)
        all_phases[muscle] = detect_doublePhases(envelope, t1, t2)

    plot_onset(emg_data, all_phases, t_signal, "Double Threshold RMS")


def onsetDetection_doubleThresholdTKEO(file_path, t1_std, t2_std):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_tkeo(signal, window_samples)
        t1, t2             = compute_doubleThreshold(envelope, t1_std, t2_std)
        all_phases[muscle] = detect_doublePhases(envelope, t1, t2)

    plot_onset(emg_data, all_phases, t_signal, "Double Threshold TKEO")


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
    #onsetDetection_doubleThresholdRMS(FILE_PATH, t1_std=5, t2_std=2)
    #onsetDetection_doubleThresholdTKEO(FILE_PATH, t1_std=5, t2_std=2)

    # Aufruf — mit Schiebereglern zum Ausprobieren
    emg_data = readC3D(FILE_PATH)
    t_signal = np.arange(len(list(emg_data.values())[0])) / ANALOG_RATE
    plot_doubleThreshold_slider(emg_data, t_signal)
