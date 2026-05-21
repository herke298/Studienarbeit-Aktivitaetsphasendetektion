import numpy as np

BASELINE_S       = 1.0
ANALOG_RATE      = 2000
baseline_samples = int(BASELINE_S * ANALOG_RATE)


"""
finde ruhigste Periode und gib Mittelwert und Standardabweichung zurück
"""
def compute_baseline_stats(envelope):
    min_var    = np.inf #damit erste Varianz zu 100% kleiner ist
    best_start = 0
    for i in range(0, len(envelope) - baseline_samples + 1, baseline_samples // 2):#baseline_samples//2 = Schrittweite
        var = np.var(envelope[i:i+baseline_samples])
        if var < min_var:
            min_var    = var
            best_start = i
    baseline = envelope[best_start:best_start + baseline_samples]
    return np.mean(baseline), np.std(baseline)
    #mean = Mittelwert, Std = Standardabweichung

"""
finde den maximalen Wert im Signal, bilde ein Fenster drum herum und berechne daraus H1 (Aktivitätsverteilung)
"""
def compute_activity_stats(envelope):
    max_idx    = np.argmax(envelope)
    start      = max(0, max_idx - baseline_samples // 2)
    #max(0,...) damit falls max_idx direkt nachm start kommt keine negative Zahl zustande kommt
    end        = min(len(envelope), max_idx + baseline_samples // 2)
    #hier das gleiche Prinzip nur eben am Ende
    activity   = envelope[start:end]
    return np.mean(activity), np.std(activity)


"""
berechne Log-Likelihood-Ratio für jedes gleitende Fenster
LLR > 0: Fenster passt besser zu H1 (Aktivität)
LLR < 0: Fenster passt besser zu H0 (Ruhe)
"""
def compute_llr(envelope, mean0, std0, mean1, std1, window):
    llr_values = []
    for i in range(len(envelope) - window + 1):
        ausschnitt = envelope[i:i+window]
        #Gauß'sche Normalverteilung 
        #log, da entspannter als e hoch...
        log_h0     = -np.log(std0) - (ausschnitt - mean0)**2 / (2 * std0**2)
        log_h1     = -np.log(std1) - (ausschnitt - mean1)**2 / (2 * std1**2)
        llr        = np.sum(log_h1 - log_h0)
        llr_values.append(llr)
    return np.array(llr_values)


"""
finde Aktivitätsphasen anhand des LLR-Signals
Onset wenn LLR über onset_threshold, Offset wenn LLR unter offset_threshold fällt
"""
def detect_llrPhases(llr, onset_threshold=0, offset_threshold=0, min_dauer_ms=2000):
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(llr)):
        if llr[i] > onset_threshold and not in_phase:
            onset    = i
            in_phase = True
        elif llr[i] <= offset_threshold and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    if in_phase:
        phases.append((onset, len(llr) - 1))

    gefiltert = []
    for onset, offset in phases:
        if (offset - onset) / ANALOG_RATE * 1000 >= min_dauer_ms:
            gefiltert.append((onset, offset))

    return gefiltert


"""
berechne Bayes-Score: LLR verschoben um den Prior
prior_activity: Vorabwahrscheinlichkeit dass eine Phase Aktivität ist (z.B. 0.2 = 20%)
"""
def compute_bayes(llr, prior_activity=0.2):
    prior_rest  = 1 - prior_activity
    bayes_score = llr + np.log(prior_activity / prior_rest)
    # wenn rest>activity --> log < 0, Wahrscheinlichkeit verschiebt sich hin zu H0
    return bayes_score


"""
berechne CUSUM über das Envelope-Signal
k: Slack-Parameter, dämpft kleine Abweichungen
"""
def compute_cusum(envelope, mean, std, k=0.5):
    cusum = np.zeros(len(envelope))
    for i in range(1, len(envelope)):#1 als Startpunkt da i-1
        cusum[i] = max(0.0, cusum[i-1] + (envelope[i] - mean) / std - k)#max(0.0,...) um negative Zahlen zu verhindern
    return cusum


"""
finde Aktivitätsphasen im CUSUM-Signal mit Envelope-Fenstervergleich für Offset-Erkennung
Onset wenn CUSUM über onset_threshold
Offset wenn aktuelles Envelope-Fenster um mehr als drop_fraction kleiner ist als vorheriges
"""
def detect_cusumPhases_diff(cusum, envelope, mean, std, onset_threshold, drop_fraction=0.3, compare_window_ms=200, k=0.5, min_dauer_ms=2000):
    compare_window = int(compare_window_ms / 1000 * ANALOG_RATE)
    phases         = []
    in_phase       = False
    onset          = 0

    for i in range(len(cusum)):
        if cusum[i] > onset_threshold and not in_phase:
            onset    = i
            in_phase = True

        elif in_phase and i >= 2 * compare_window:
            prev_window = envelope[i - 2*compare_window : i - compare_window]
            curr_window = envelope[i - compare_window : i]
            prev_score  = np.mean((prev_window - mean) / std - k)
            curr_score  = np.mean((curr_window - mean) / std - k)
            if curr_score < drop_fraction * prev_score:
                offset   = i
                in_phase = False
                phases.append((onset, offset))

    if in_phase:
        phases.append((onset, len(cusum) - 1))

    gefiltert = []
    for onset, offset in phases:
        if (offset - onset) / ANALOG_RATE * 1000 >= min_dauer_ms:
            gefiltert.append((onset, offset))

    return gefiltert


"""
finde Aktivitätsphasen im CUSUM-Signal
Onset wenn CUSUM über onset_threshold, Offset wenn CUSUM unter offset_threshold fällt
"""
def detect_cusumPhases(cusum, onset_threshold, offset_threshold=0, min_dauer_ms=2000):
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(cusum)):
        if cusum[i] > onset_threshold and not in_phase:
            onset    = i
            in_phase = True
        elif cusum[i] <= offset_threshold and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    if in_phase:
        phases.append((onset, len(cusum) - 1))

    gefiltert = []
    for onset, offset in phases:
        if (offset - onset) / ANALOG_RATE * 1000 >= min_dauer_ms:
            gefiltert.append((onset, offset))

    return gefiltert
