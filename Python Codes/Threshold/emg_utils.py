import numpy as np
from onset_utils import compute_rms, compute_tkeo

BASELINE_S   = 1.0
ANALOG_RATE  = 2000
baseline_samples = int(BASELINE_S * ANALOG_RATE)


"""
finde die Aktivitätsphasen
"""
def detect_phases(signal, threshold, min_dauer_ms=2000):
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(signal)):

        #wenn Wert über Threshold ist und noch nicht erkannt wurde, setze den onset
        if signal[i] > threshold and not in_phase:
            onset    = i
            in_phase = True

        #wenn wert unter Threshold ist und sich in einer Aktivitätsphase befindet setze den offset
        elif signal[i] <= threshold and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    #Sonderfall letztes Signal ist immer noch Aktivitätsphase
    if in_phase:
        phases.append((onset, len(signal) - 1))

    #nur Phasen behalten die länger als min_dauer sind
    gefiltert = []
    for onset, offset in phases:
        if (offset - onset) / ANALOG_RATE * 1000 >= min_dauer_ms:
            gefiltert.append((onset, offset))

    return gefiltert


"""
finde ruhigste Periode und berechne daraus den Threshold
"""
def compute_threshold(signal, threshold_std):
    min_var    = np.inf
    best_start = 0
    for i in range(0, len(signal) - baseline_samples + 1, baseline_samples // 2): #//:Ganzzahldivision
        var = np.var(signal[i:i+baseline_samples]) #bilde Varianz
        if var < min_var:                        #falls es kleinste Varianz war, setze sie
            min_var    = var
            best_start = i

    baseline  = signal[best_start:best_start + baseline_samples]
    threshold = np.mean(baseline) + threshold_std * np.std(baseline) #berechne den Threshold
    return threshold


"""
finde ruhigste Periode und berechne daraus zwei Thresholds:
T1 (hoch) für Onset-Erkennung, T2 (niedrig) für Offset-Erkennung
"""
def compute_doubleThreshold(signal, t1_std, t2_std):
    min_var    = np.inf
    best_start = 0
    for i in range(0, len(signal) - baseline_samples + 1, baseline_samples // 2): #//:Ganzzahldivision
        var = np.var(signal[i:i+baseline_samples]) #bilde Varianz
        if var < min_var:                        #falls es kleinste Varianz war, setze sie
            min_var    = var
            best_start = i

    baseline = signal[best_start:best_start + baseline_samples]
    t1       = np.mean(baseline) + t1_std * np.std(baseline) #hoher Threshold für Onset
    t2       = np.mean(baseline) + t2_std * np.std(baseline) #niedriger Threshold für Offset
    return t1, t2


"""
finde Aktivitätsphasen mit zwei Schwellen:
Onset wenn Signal über T1, Offset erst wenn Signal unter T2
"""
def detect_doublePhases(signal, t1, t2, min_dauer_ms=2000):
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(signal)):

        #wenn Wert über T1 und noch nicht in Phase, setze Onset
        if signal[i] > t1 and not in_phase:
            onset    = i
            in_phase = True

        #wenn Wert unter T2 und in Phase, setze Offset
        elif signal[i] < t2 and in_phase:
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
