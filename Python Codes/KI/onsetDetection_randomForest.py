import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import numpy as np
import joblib
from read_c3d import readC3D
from plot_onset import plot_onset
from emg_utils_ki import extract_features, filter_active_muscles

# Parameter
ANALOG_RATE = 2000
WINDOW_MS   = 50
window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE)

MODEL_PATH = r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\KI\Model\randomForest.pkl"


"""
wandle binäre Vorhersagen (0/1 pro Fenster) in Onset/Offset-Paare um
"""
def predictions_to_phases(predictions, min_dauer_ms=2000, analog_rate=ANALOG_RATE):
    min_dauer_samples = int(min_dauer_ms / 1000 * analog_rate)
    phases   = []
    in_phase = False
    onset    = 0

    for i in range(len(predictions)):
        if predictions[i] == 1 and not in_phase:
            onset    = i
            in_phase = True
        elif predictions[i] == 0 and in_phase:
            offset   = i
            in_phase = False
            phases.append((onset, offset))

    if in_phase:
        phases.append((onset, len(predictions) - 1))

    gefiltert = []
    for onset, offset in phases:
        if (offset - onset) >= min_dauer_samples:
            gefiltert.append((onset, offset))

    return gefiltert


def onsetDetection_randomForest(file_path, model_path=MODEL_PATH, min_dauer_ms=2000, schwelle=0.3):
    model          = joblib.load(model_path)
    emg_data       = readC3D(file_path)
    file_name      = file_path.split('\\')[-1].replace('.c3d', '')
    muscles        = filter_active_muscles(list(emg_data.keys()), file_name)
    all_phases     = {}

    for muscle in muscles:
        signal             = emg_data[muscle]
        t_signal           = np.arange(len(signal)) / ANALOG_RATE
        features              = extract_features(signal)
        wahrscheinlichkeiten  = model.predict_proba(features)[:, 1]
        predictions           = (wahrscheinlichkeiten >= schwelle).astype(int)
        all_phases[muscle]    = predictions_to_phases(predictions, min_dauer_ms)

    plot_onset(emg_data, all_phases, t_signal, "Random Forest")


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\BS1507\BS1507_Isokinetic_Right_2025_07_15.c3d"
    onsetDetection_randomForest(FILE_PATH, min_dauer_ms=2000)
