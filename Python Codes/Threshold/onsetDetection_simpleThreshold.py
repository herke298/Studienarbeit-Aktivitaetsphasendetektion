import sys
sys.path.append(r"C:\Studienarbeit\Arbeitsordner\Python Codes")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from emg_utils import compute_rms, detect_phases

# Parameter
ANALOG_RATE  = 2000
WINDOW_MS    = 50
BASELINE_S   = 1.0

window_samples   = int(WINDOW_MS / 1000 * ANALOG_RATE) #rechnet Fenstergröße von ms in Samples um
baseline_samples = int(BASELINE_S * ANALOG_RATE)


"""
finde ruhigste Periode und berechne daraus den Threshold
"""
def compute_threshold(rms, threshold_std):
    min_var    = np.inf
    best_start = 0
    for i in range(0, len(rms) - baseline_samples + 1, baseline_samples // 2): #//:Ganzzahldivision
        var = np.var(rms[i:i+baseline_samples]) #bilde Varianz
        if var < min_var:                        #falls es kleinste Varianz war, setze sie
            min_var    = var
            best_start = i

    baseline  = rms[best_start:best_start + baseline_samples]
    threshold = np.mean(baseline) + threshold_std * np.std(baseline) #berechne den Threshold
    return threshold


"""
Hauptfunktion
"""
def onsetDetection_simpleThreshold(file_path, threshold_std):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE  #Zeitachse für Rohsignal
        rms                = compute_rms(signal, window_samples)
        threshold          = compute_threshold(rms, threshold_std)
        all_phases[muscle] = detect_phases(rms, threshold)

    plot_onset(emg_data, all_phases, t_signal, "Simple Threshold")


# Aufruf
FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
onsetDetection_simpleThreshold(FILE_PATH, threshold_std=5)
