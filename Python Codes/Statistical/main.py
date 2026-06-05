import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")

from onsetDetection_cusum import onsetDetection_cusumRMS,      onsetDetection_cusumTKEO
from onsetDetection_cusum import onsetDetection_cusumDiffRMS,  onsetDetection_cusumDiffTKEO
from onsetDetection_llr   import onsetDetection_llrRMS,        onsetDetection_llrTKEO
from onsetDetection_llr   import onsetDetection_bayesRMS,      onsetDetection_bayesTKEO

FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"

#onsetDetection_cusumRMS(FILE_PATH,      onset_threshold=5,  min_dauer_ms=2000)
#onsetDetection_cusumTKEO(FILE_PATH,     onset_threshold=5,  min_dauer_ms=2000)

#onsetDetection_cusumDiffRMS(FILE_PATH,  onset_threshold=5,  drop_fraction=0.3, min_dauer_ms=2000)
#onsetDetection_cusumDiffTKEO(FILE_PATH, onset_threshold=5,  drop_fraction=0.3, min_dauer_ms=2000)

#onsetDetection_llrRMS(FILE_PATH,        onset_threshold=0,  min_dauer_ms=2000)
onsetDetection_llrTKEO(FILE_PATH,       onset_threshold=0,  min_dauer_ms=2000)

#onsetDetection_bayesRMS(FILE_PATH,      prior_activity=0.2, min_dauer_ms=2000)
#onsetDetection_bayesTKEO(FILE_PATH,     prior_activity=0.2, min_dauer_ms=2000)
