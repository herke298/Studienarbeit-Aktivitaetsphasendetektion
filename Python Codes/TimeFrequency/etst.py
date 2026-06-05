import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\Statistical")

from onset_utils import compute_wavelet_energy
from emg_utils_stat import compute_baseline_stats, compute_activity_stats
from read_c3d import readC3D
import numpy as np

FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"

emg_data  = readC3D(FILE_PATH)
signal    = list(emg_data.values())[0]
envelope  = compute_wavelet_energy(signal, 2000)

mean0, std0 = compute_baseline_stats(envelope)
mean1, std1 = compute_activity_stats(envelope)

print("Baseline (Ruhe):", mean0)
print("Aktivität:      ", mean1)
print("Threshold 0.5:  ", mean0 + 0.5 * (mean1 - mean0))
print("Max Envelope:   ", np.max(envelope))