import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

import os
import json
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from read_c3d import readC3D
from emg_utils_ki import extract_features_wamp, label_windows, filter_active_muscles

# Pfade
LABEL_DIR  = r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\Labeling\Activity"
DATA_DIR   = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten"
MODEL_PATH = r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\KI\Model\lgbm.pkl"


def train(label_dir=LABEL_DIR, data_dir=DATA_DIR, model_path=MODEL_PATH):
    alle_features = []
    alle_labels   = []

    """lade alle JSON-Labeldateien"""
    for json_file in os.listdir(label_dir):
        if not json_file.endswith('_labels.json'):
            continue

        json_path = os.path.join(label_dir, json_file)
        file_name = json_file.replace('_labels.json', '')

        """suche zugehörige C3D-Datei"""
        c3d_path = None
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.replace('.c3d', '') == file_name:
                    c3d_path = os.path.join(root, f)
                    break

        if c3d_path is None:
            print(f"C3D-Datei nicht gefunden für: {file_name}")
            continue

        print(f"Verarbeite: {file_name}")

        with open(json_path, 'r') as f:
            labels_data = json.load(f)

        emg_data       = readC3D(c3d_path)
        aktive_muskeln = filter_active_muscles(list(emg_data.keys()), file_name)

        for muscle, phases in labels_data[file_name].items():
            if muscle not in emg_data:
                continue
            if muscle not in aktive_muskeln:
                continue

            signal   = emg_data[muscle]
            features = extract_features_wamp(signal)
            labels   = label_windows(len(features), phases)

            alle_features.append(features)
            alle_labels.append(labels)

    if len(alle_features) == 0:
        print("Keine Trainingsdaten gefunden!")
        return

    """kombiniere alle Features und Labels"""
    X = np.vstack(alle_features)
    y = np.concatenate(alle_labels)

    print(f"\nTrainingsdaten: {len(X)} Fenster")
    print(f"Aktivität: {np.sum(y == 1)} Fenster ({100*np.mean(y):.1f}%)")
    print(f"Ruhe:      {np.sum(y == 0)} Fenster ({100*(1-np.mean(y)):.1f}%)")

    """begrenze auf maximal 50000 Fenster pro Klasse"""
    max_pro_klasse = 50000
    idx_ruhe       = np.where(y == 0)[0]
    idx_aktiv      = np.where(y == 1)[0]

    if len(idx_ruhe) > max_pro_klasse:
        idx_ruhe = np.random.choice(idx_ruhe, max_pro_klasse, replace=False)
    if len(idx_aktiv) > max_pro_klasse:
        idx_aktiv = np.random.choice(idx_aktiv, max_pro_klasse, replace=False)

    idx_gesamt = np.concatenate([idx_ruhe, idx_aktiv])
    X = X[idx_gesamt]
    y = y[idx_gesamt]

    """aufteilen in Training und Test"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """LGBM mit Normalisierung trainieren"""
    print("\nTrainiere LGBM...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lgbm',   LGBMClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)

    """Ergebnis ausgeben"""
    y_pred = model.predict(X_test)
    print("\nErgebnis auf Testdaten:")
    print(classification_report(y_test, y_pred, target_names=['Ruhe', 'Aktivität']))

    """Modell speichern"""
    joblib.dump(model, model_path)
    print(f"Modell gespeichert: {model_path}")


if __name__ == "__main__":
    train()
