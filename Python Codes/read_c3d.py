import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# Pfad zur C3D-Datei anpassen

def readC3D(file_path):
    # Datei einlesen
    c = ezc3d.c3d(file_path)

    params        = c["parameters"]
    analogs       = c["data"]["analogs"][0]
    analog_labels = params["ANALOG"]["LABELS"]["value"]
    analog_units  = params["ANALOG"].get("UNITS", {}).get("value", [""] * len(analog_labels))
    analog_rate   = params["ANALOG"]["RATE"]["value"][0]

    print(file_path)
    print(f"Dauer: {analogs.shape[1] / analog_rate:.1f}s")

    t = np.arange(analogs.shape[1]) / analog_rate

    # Nur EMG-Kanäle
    emg_indices = [i for i, u in enumerate(analog_units) if "V" in u]
    emg_labels  = [analog_labels[i] for i in emg_indices]
    return {emg_labels[j]: analogs[idx] for j, idx in enumerate(emg_indices)}

r"""
FILE_PATH1 = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
readC3D(FILE_PATH1)
"""