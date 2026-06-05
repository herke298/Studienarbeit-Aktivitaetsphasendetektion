import numpy as np


ANALOG_RATE    = 2000
WINDOW_MS      = 50
window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE)


"""
extrahiere Features für jedes gleitende Fenster im Signal
Features: RMS, TKEO, STFT-Energie, Zero-Crossing-Rate, Varianz
"""
def extract_features(signal, analog_rate=ANALOG_RATE, window=window_samples):
    features = []

    for i in range(len(signal) - window + 1):
        ausschnitt = signal[i:i+window]

        """RMS"""
        rms = np.sqrt(np.mean(ausschnitt**2))

        """TKEO"""
        tkeo      = ausschnitt[1:-1]**2 - ausschnitt[:-2] * ausschnitt[2:]
        tkeo_mean = np.mean(np.abs(tkeo))

        """STFT-Energie (20-500 Hz)"""
        fft_vals    = np.fft.rfft(ausschnitt)
        freqs       = np.fft.rfftfreq(window, d=1/analog_rate)
        idx         = (freqs >= 20) & (freqs <= 500)
        stft_energy = np.sum(np.abs(fft_vals[idx])**2) / window

        """Zero-Crossing-Rate"""
        zcr = np.sum(np.diff(np.sign(ausschnitt)) != 0) / window

        """Varianz"""
        var = np.var(ausschnitt)

        features.append([rms, tkeo_mean, stft_energy, zcr, var])

    return np.array(features)


"""
erkenne anhand des Dateinamens welche Seite aktiv ist (Left/Right)
und filtere nur die Muskeln der aktiven Seite
"""
def filter_active_muscles(muscles, file_name):
    file_name_lower = file_name.lower()

    if 'left' in file_name_lower:
        return [m for m in muscles if m.lower().startswith('l.') or m.lower().startswith('l_') or 'left' in m.lower()]
    elif 'right' in file_name_lower:
        return [m for m in muscles if m.lower().startswith('r.') or m.lower().startswith('r_') or 'right' in m.lower()]
    else:
        return muscles


"""
berechne WAMP (Willison Amplitude) für jedes gleitende Fenster
WAMP zählt wie oft die Amplitudendifferenz zwischen benachbarten Samples den Threshold überschreitet
Threshold = Mittelwert des gesamten Signals (laut Paper)
"""
def extract_features_wamp(signal, analog_rate=ANALOG_RATE, window=window_samples):
    threshold = np.mean(np.abs(signal))
    features  = []

    for i in range(len(signal) - window + 1):
        ausschnitt = signal[i:i+window]
        wamp       = np.sum(np.abs(np.diff(ausschnitt)) >= threshold)
        features.append([wamp])

    return np.array(features)


"""
weise jedem Fenster ein Label zu (0=Ruhe, 1=Aktivität)
basierend auf Onset/Offset-Paaren aus den Labeling-Daten
"""
def label_windows(n_windows, phases_s, analog_rate=ANALOG_RATE, window=window_samples):
    labels = np.zeros(n_windows, dtype=int)

    for onset_s, offset_s in phases_s:
        onset_idx  = int(onset_s  * analog_rate)
        offset_idx = int(offset_s * analog_rate)

        for i in range(n_windows):
            mitte = i + window // 2
            if onset_idx <= mitte <= offset_idx:
                labels[i] = 1

    return labels
