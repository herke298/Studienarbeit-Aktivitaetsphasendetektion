import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import numpy as np
from read_c3d import readC3D
from plot_onset import plot_onset
from emg_utils import compute_signal, compute_threshold

# Parameter
ANALOG_RATE    = 2000
WINDOW_MS      = 50
ADAPT_WINDOW_S = 2.0  # Größe des Fensters für die adaptive Threshold-Aktualisierung

window_samples = int(WINDOW_MS      / 1000 * ANALOG_RATE) #rechnet Fenstergröße von ms in Samples um
adapt_samples  = int(ADAPT_WINDOW_S * ANALOG_RATE)


"""
Adaptive Threshold mit Freeze-Ansatz:
- in Ruhephasen wird der Threshold aus dem gleitenden Fenster neu berechnet
- in Aktivitätsphasen wird der Threshold eingefroren
"""
def detect_adaptivePhases(signal, threshold_std, min_dauer_ms=2000):
    phases    = []
    in_phase  = False
    onset     = 0

    #initialer Threshold aus ruhigster Periode
    threshold = compute_threshold(signal, threshold_std)

    for i in range(len(signal)):
        #wenn nicht in Phase und mindestens schon ein Fenster durchgelaufen ist, berechne threshold vom letzten Fensterausschnitt
        if not in_phase and i >= adapt_samples:
            fenster   = signal[i - adapt_samples:i]                        #letztes Fenster
            threshold = np.mean(fenster) + threshold_std * np.std(fenster) #Threshold aktualisieren

        #wenn Wert über Threshold und noch nicht in Phase, setze Onset (Threshold wird nicht neu berechnet)
        if signal[i] > threshold and not in_phase:
            onset    = i
            in_phase = True

        #wenn Wert unter Threshold und in Phase, setze Offset (Threshold wird wieder aktualisiert)
        elif signal[i] <= threshold and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    #Sonderfall: letztes Signal ist immer noch Aktivitätsphase
    if in_phase:
        phases.append((onset, len(signal) - 1))

    #nur Phasen behalten die länger als min_dauer sind
    gefiltert = []
    for onset, offset in phases:
        if (offset - onset) / ANALOG_RATE * 1000 >= min_dauer_ms:
            gefiltert.append((onset, offset))

    return gefiltert


"""
Hauptfunktion
"""
def onsetDetection_adaptiveThreshold(file_path, threshold_std, min_dauer_ms=2000):
    emg_data   = readC3D(file_path)
    muscles    = list(emg_data.keys())
    all_phases = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE  #Zeitachse für Rohsignal
        signal                = compute_rms(signal, window_samples)
        all_phases[muscle] = detect_adaptivePhases(signal, threshold_std, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Adaptive Threshold")


# Aufruf
FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
onsetDetection_adaptiveThreshold(FILE_PATH, threshold_std=3, min_dauer_ms=2000)
