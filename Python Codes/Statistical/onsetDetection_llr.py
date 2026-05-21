import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from onset_utils import compute_rms, compute_tkeo
from emg_utils_stat import compute_baseline_stats, compute_activity_stats, compute_llr, detect_llrPhases, compute_bayes

# Parameter
ANALOG_RATE    = 2000
WINDOW_MS      = 50

window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE)


def onsetDetection_llrRMS(file_path, onset_threshold=0, offset_threshold=0, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_rms(signal, window_samples)
        mean0, std0        = compute_baseline_stats(envelope)
        mean1, std1        = compute_activity_stats(envelope)
        llr                = compute_llr(envelope, mean0, std0, mean1, std1, window_samples)
        all_phases[muscle] = detect_llrPhases(llr, onset_threshold, offset_threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Likelihood-Ratio RMS")


def onsetDetection_llrTKEO(file_path, onset_threshold=0, offset_threshold=0, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_tkeo(signal, window_samples)
        mean0, std0        = compute_baseline_stats(envelope)
        mean1, std1        = compute_activity_stats(envelope)
        llr                = compute_llr(envelope, mean0, std0, mean1, std1, window_samples)
        all_phases[muscle] = detect_llrPhases(llr, onset_threshold, offset_threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Likelihood-Ratio TKEO")


def onsetDetection_bayesRMS(file_path, prior_activity=0.2, onset_threshold=0, offset_threshold=0, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_rms(signal, window_samples)
        mean0, std0        = compute_baseline_stats(envelope)
        mean1, std1        = compute_activity_stats(envelope)
        llr                = compute_llr(envelope, mean0, std0, mean1, std1, window_samples)
        bayes_score        = compute_bayes(llr, prior_activity)
        all_phases[muscle] = detect_llrPhases(bayes_score, onset_threshold, offset_threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Bayes RMS")


def onsetDetection_bayesTKEO(file_path, prior_activity=0.2, onset_threshold=0, offset_threshold=0, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        envelope           = compute_tkeo(signal, window_samples)
        mean0, std0        = compute_baseline_stats(envelope)
        mean1, std1        = compute_activity_stats(envelope)
        llr                = compute_llr(envelope, mean0, std0, mean1, std1, window_samples)
        bayes_score        = compute_bayes(llr, prior_activity)
        all_phases[muscle] = detect_llrPhases(bayes_score, onset_threshold, offset_threshold, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Bayes TKEO")


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
    #onsetDetection_llrRMS(FILE_PATH,   onset_threshold=0, offset_threshold=0, min_dauer_ms=2000)
    #onsetDetection_llrTKEO(FILE_PATH,  onset_threshold=0, offset_threshold=0, min_dauer_ms=2000)
    #onsetDetection_bayesRMS(FILE_PATH,  prior_activity=0.2, min_dauer_ms=2000)
    onsetDetection_bayesTKEO(FILE_PATH, prior_activity=0.2, min_dauer_ms=2000)
