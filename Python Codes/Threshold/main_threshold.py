import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

from onsetDetection_simpleThreshold   import onsetDetection_simpleThresholdRMS,   onsetDetection_simpleThresholdTKEO
from onsetDetection_doubleThreshold   import onsetDetection_doubleThresholdRMS,   onsetDetection_doubleThresholdTKEO
from onsetDetection_adaptiveThreshold import onsetDetection_adaptiveThresholdRMS, onsetDetection_adaptiveThresholdTKEO

FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"

onsetDetection_simpleThresholdRMS(FILE_PATH,   threshold_std=5)
onsetDetection_simpleThresholdTKEO(FILE_PATH,  threshold_std=5)

onsetDetection_doubleThresholdRMS(FILE_PATH,   t1_std=5, t2_std=2)
onsetDetection_doubleThresholdTKEO(FILE_PATH,  t1_std=5, t2_std=2)

onsetDetection_adaptiveThresholdRMS(FILE_PATH,  threshold_std=3, min_dauer_ms=50)
onsetDetection_adaptiveThresholdTKEO(FILE_PATH, threshold_std=3, min_dauer_ms=50)
