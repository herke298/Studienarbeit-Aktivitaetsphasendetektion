import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from onset_utils import compute_rms, compute_tkeo
from emg_utils_stat import compute_baseline_stats, compute_cusum, detect_cusumPhases, detect_cusumPhases_diff

# Parameter
ANALOG_RATE    = 2000
WINDOW_MS      = 50

window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE)


def onsetDetection_cusumRMS(file_path, onset_threshold=5, offset_threshold=0, k=0.5, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_rms(signal, window_samples)
        mean, std          = compute_baseline_stats(envelope)
        cusum              = compute_cusum(envelope, mean, std, k)
        all_phases[muscle] = detect_cusumPhases(cusum, onset_threshold, offset_threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "CUSUM RMS")


def onsetDetection_cusumTKEO(file_path, onset_threshold=400, offset_threshold=100, k=0.5, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_tkeo(signal, window_samples)
        mean, std          = compute_baseline_stats(envelope)
        cusum              = compute_cusum(envelope, mean, std, k)
        all_phases[muscle] = detect_cusumPhases(cusum, onset_threshold, offset_threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "CUSUM TKEO")


def onsetDetection_cusumDiffRMS(file_path, onset_threshold=5, drop_fraction=0.3, compare_window_ms=200, k=0.5, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_rms(signal, window_samples)
        mean, std          = compute_baseline_stats(envelope)
        cusum              = compute_cusum(envelope, mean, std, k)
        all_phases[muscle] = detect_cusumPhases_diff(cusum, envelope, mean, std, onset_threshold, drop_fraction, compare_window_ms, k, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "CUSUM Diff RMS")


def onsetDetection_cusumDiffTKEO(file_path, onset_threshold=5, drop_fraction=0.3, compare_window_ms=200, k=0.5, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_tkeo(signal, window_samples)
        mean, std          = compute_baseline_stats(envelope)
        cusum              = compute_cusum(envelope, mean, std, k)
        all_phases[muscle] = detect_cusumPhases_diff(cusum, envelope, mean, std, onset_threshold, drop_fraction, compare_window_ms, k, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "CUSUM Diff TKEO")


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
    #onsetDetection_cusumRMS(FILE_PATH,  onset_threshold=5, offset_threshold=0, k=0.5, min_dauer_ms=20000)
    #onsetDetection_cusumTKEO(FILE_PATH, onset_threshold=5, offset_threshold=0, k=0.5, min_dauer_ms=20000)
    onsetDetection_cusumDiffRMS(FILE_PATH,onset_threshold=1000,drop_fraction=0.1,k=0.5,min_dauer_ms=2000)
