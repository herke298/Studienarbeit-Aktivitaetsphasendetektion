import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\Threshold")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from onset_utils import compute_wavelet_energy
from emg_utils import detect_phases, compute_threshold

# Parameter
ANALOG_RATE = 2000


def onsetDetection_wavelet(file_path, threshold_std=3, wavelet='db4', level=5, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_wavelet_energy(signal, ANALOG_RATE, wavelet, level)
        threshold          = compute_threshold(envelope, threshold_std)
        all_phases[muscle] = detect_phases(envelope, threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Wavelet")


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
    onsetDetection_wavelet(FILE_PATH, threshold_std=3, min_dauer_ms=2000)
