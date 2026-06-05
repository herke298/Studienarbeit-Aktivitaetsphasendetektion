import numpy as np
import pywt


"""
------------------------------------------------------
schiebe Fenster über Signal, und bilde in jedem Fenster quadratischen Mittelwert(rms)
------------------------------------------------------
"""
def compute_rms(signal, window):
    rms_values = []
    """iteriere über gesamte Länge des Signals"""
    for i in range(len(signal) - window + 1): #len(signal)-window+1 verhindert, dass Fenster übers Ende hinaus geht

        """berechne aktuellen Ausschnitt"""
        ausschnitt = signal[i:i+window]        #Fenster übers Ende hinaus geht

        """quadriere Ausschnitt"""
        quadriert  = ausschnitt ** 2           #+1 bewirkt 99%ige Überlappung

        """bilde Mittelwert vom Ausschnitt"""
        mittelwert = np.mean(quadriert)

        """quadriere Ausschnitt"""
        rms        = np.sqrt(mittelwert)

        """füge Rms-Werte dem Array hinzu"""
        rms_values.append(rms)
        
    return np.array(rms_values)


"""
-------------------------------------------------------------
schiebe Fenster über Signal, berechne FFT und summiere  Energie im EMG-Frequenzbereich (f_low(20Hz) bis f_high(500HZ))
-------------------------------------------------------------
"""
def compute_stft_energy(signal, window, analog_rate, f_low=20, f_high=500):
    energy_values = []

    """iteriere über gesamte Länge des Signals"""
    for i in range(len(signal) - window + 1):
        """berechne aktuellen Ausschnitt"""
        ausschnitt = signal[i:i+window]

        """berechne die Werte mittels FFT"""
        fft_vals   = np.fft.rfft(ausschnitt)#rfft, da nur reelle Werte vorhanden -->spart Rechenzeit und Speicher

        """bestimme die Frequenzanteile"""
        freqs      = np.fft.rfftfreq(window, d=1/analog_rate)#window=Anzahl Samples im Fenster, d=ANzahl Frequenzschritte

        """filtere Frequenzen unter 20Hz und über 500Hz raus"""
        idx        = (freqs >= f_low) & (freqs <= f_high)#filtert ungewollte Frequenzen raus(zB. Elektrorauschen >500Hz, Bewegungsartefakte < 200Hz)
       
        """berechne die Energie des Abschnitts"""
        energie    = np.sum(np.abs(fft_vals[idx])**2) / window #/window um energie von Fenstergöße unabhängig zu machen
        
        """füge Energie-Werte dem Array hinzu"""
        energy_values.append(energie)

    return np.array(energy_values)


"""
---------------------------------------------------------------
berechne Wavelet-Energie mittels CWT und Morlet-Wavelet
Morlet peakt in der Mitte → Artefakte am Signalanfang/-ende haben geringere Ähnlichkeit
summiere Energie über EMG-relevante Frequenzen (f_low bis f_high Hz)
---------------------------------------------------------------
"""
def compute_wavelet_energy(signal, analog_rate, wavelet='morl', f_low=20, f_high=500):

    """berechne Skalen für den EMG-Frequenzbereich"""
    dt          = 1 / analog_rate
    center_freq = pywt.central_frequency(wavelet)
    scale_min   = int(center_freq / (f_high * dt))
    scale_max   = int(center_freq / (f_low  * dt))
    scales      = np.arange(max(1, scale_min), scale_max + 1)

    """berechne CWT — gibt Koeffizienten in Originallänge zurück"""
    coeffs, _   = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    """summiere Energie über alle relevanten Skalen"""
    energy      = np.sum(np.abs(coeffs) ** 2, axis=0)

    return energy


"""
---------------------------------------------------------------
schiebe ein Fenster über das Signal, und bilde in jedem Fenster den TKEO-Mittelwert
---------------------------------------------------------------
"""
def compute_tkeo(signal, window):
    tkeo_values = []

    """iteriere über die gesamte Länge des Signals"""
    for i in range(len(signal) - window + 1):

        """berechne aktuellen Ausschnitt"""
        ausschnitt = signal[i:i+window]

        """berechne tkeo des aktuellen Ausschnitts"""
        tkeo       = ausschnitt[1:-1]**2 - ausschnitt[:-2] * ausschnitt[2:]

        """Füge den Mittelwert der Beträge vom TKEO des Ausschnitts dem Array hinzu"""
        tkeo_values.append(np.mean(np.abs(tkeo)))

    return np.array(tkeo_values)
