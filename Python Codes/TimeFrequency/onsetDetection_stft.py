import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\Threshold")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from onset_utils import compute_stft_energy
from emg_utils import compute_threshold, detect_phases

# Parameter
ANALOG_RATE    = 2000
WINDOW_MS      = 50
F_LOW          = 20
F_HIGH         = 500

window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE)


def onsetDetection_stft(file_path, threshold_std=3, f_low=F_LOW, f_high=F_HIGH, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_stft_energy(signal, window_samples, ANALOG_RATE, f_low, f_high)
        threshold          = compute_threshold(envelope, threshold_std)
        all_phases[muscle] = detect_phases(envelope, threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "STFT")


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isokinetic_Right_2025_05_08.c3d"
    onsetDetection_stft(FILE_PATH, threshold_std=3, min_dauer_ms=2000)
