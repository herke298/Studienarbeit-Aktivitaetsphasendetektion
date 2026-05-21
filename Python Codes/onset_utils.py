import numpy as np


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
schiebe ein Fenster über das Signal, und bilde in jedem Fenster den TKEO-Mittelwert
"""
def compute_tkeo(signal, window):
    tkeo_values = []
    for i in range(len(signal) - window + 1):
        ausschnitt = signal[i:i+window]
        tkeo       = ausschnitt[1:-1]**2 - ausschnitt[:-2] * ausschnitt[2:]
        tkeo_values.append(np.mean(np.abs(tkeo)))
    return np.array(tkeo_values)
