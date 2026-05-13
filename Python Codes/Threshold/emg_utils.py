import numpy as np

BASELINE_S   = 1.0
ANALOG_RATE  = 2000
baseline_samples = int(BASELINE_S * ANALOG_RATE)


"""
schiebe ein Fenster über das Signal, und bilde in jedem Fenster den quadratischen Mittelwert(rms)
"""
def compute_rms(signal, window):
    rms_values = []
    for i in range(len(signal) - window + 1): #len(signal)-window+1 verhindert, dass
        ausschnitt = signal[i:i+window]        #Fenster übers Ende hinaus geht
        quadriert  = ausschnitt ** 2           #+1 bewirkt 99%ige Überlappung
        mittelwert = np.mean(quadriert)
        rms        = np.sqrt(mittelwert)
        rms_values.append(rms)
    return np.array(rms_values)


"""
finde die Aktivitätsphasen
"""
def detect_phases(rms, threshold):
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(rms)):

        #wenn Wert über Threshold ist und noch nicht erkannt wurde, setze den onset
        if rms[i] > threshold and not in_phase:
            onset    = i
            in_phase = True

        #wenn wert unter Threshold ist und sich in einer Aktivitätsphase befindet setze den offset
        elif rms[i] <= threshold and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    #Sonderfall letztes Signal ist immer noch Aktivitätsphase
    if in_phase:
        phases.append((onset, len(rms) - 1))

    return phases


"""
finde ruhigste Periode und berechne daraus zwei Thresholds:
T1 (hoch) für Onset-Erkennung, T2 (niedrig) für Offset-Erkennung
"""
def compute_doubleThreshold(rms, t1_std, t2_std):
    min_var    = np.inf
    best_start = 0
    for i in range(0, len(rms) - baseline_samples + 1, baseline_samples // 2): #//:Ganzzahldivision
        var = np.var(rms[i:i+baseline_samples]) #bilde Varianz
        if var < min_var:                        #falls es kleinste Varianz war, setze sie
            min_var    = var
            best_start = i

    baseline = rms[best_start:best_start + baseline_samples]
    t1       = np.mean(baseline) + t1_std * np.std(baseline) #hoher Threshold für Onset
    t2       = np.mean(baseline) + t2_std * np.std(baseline) #niedriger Threshold für Offset
    return t1, t2


"""
finde Aktivitätsphasen mit zwei Schwellen:
Onset wenn Signal über T1, Offset erst wenn Signal unter T2
"""
def detect_doublePhases(rms, t1, t2):
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(rms)):

        #wenn Wert über T1 und noch nicht in Phase, setze Onset
        if rms[i] > t1 and not in_phase:
            onset    = i
            in_phase = True

        #wenn Wert unter T2 und in Phase, setze Offset
        elif rms[i] < t2 and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    #Sonderfall: letztes Signal ist immer noch Aktivitätsphase
    if in_phase:
        phases.append((onset, len(rms) - 1))

    return phases
