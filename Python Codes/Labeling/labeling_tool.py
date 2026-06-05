import sys
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes")
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\Threshold")
sys.path.append(r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\KI")

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from read_c3d import readC3D
from onset_utils import compute_wavelet_energy
from emg_utils import detect_phases
from emg_utils_ki import filter_active_muscles

# Parameter
ANALOG_RATE = 2000


"""
berechne adaptiven Threshold zwischen ruhigstem und aktivstem Fenster
"""
def compute_adaptive_threshold(envelope, factor=0.5):
    baseline_samples = int(1.0 * ANALOG_RATE)
    min_mean         = np.inf
    max_mean         = -np.inf

    for i in range(0, len(envelope) - baseline_samples + 1, baseline_samples // 2):
        fenster = envelope[i:i+baseline_samples]
        mean    = np.mean(fenster)
        if mean < min_mean:
            min_mean = mean
        if mean > max_mean:
            max_mean = mean

    return min_mean + factor * (max_mean - min_mean)


def label_file(file_path, output_path=None):
    emg_data  = readC3D(file_path)
    muscles   = list(emg_data.keys())
    labels    = {}

    file_name = file_path.split('\\')[-1].replace('.c3d', '')

    if output_path is None:
        activity_dir = r"C:\Studienarbeit\Git\Studienarbeit-Aktivitaetsphasendetektion\Python Codes\Labeling\Activity"
        output_path  = activity_dir + '\\' + file_name + '_labels.json'

    muscles = filter_active_muscles(muscles, file_name)

    for muscle in muscles:
        signal        = emg_data[muscle]
        envelope_raw  = compute_wavelet_energy(signal, ANALOG_RATE)

        """glätte die Wavelet-Energie mit gleitendem Mittelwert (200ms Fenster)"""
        smooth_window = int(0.2 * ANALOG_RATE)
        envelope      = np.convolve(envelope_raw, np.ones(smooth_window) / smooth_window, mode='same')
        t_envelope    = np.arange(len(envelope)) / ANALOG_RATE

        """automatische Vorschlag mit adaptivem Threshold"""
        threshold   = compute_adaptive_threshold(envelope, factor=0.04)
        auto_phases = detect_phases(envelope, threshold, min_dauer_ms=1000)

        """Vorschlag in Zeitwerte umrechnen"""
        current_clicks = []
        current_phases = [(t_envelope[onset], t_envelope[offset]) for onset, offset in auto_phases]
        onset_lines    = []
        phase_patches  = []

        """erstelle Figure mit zwei Subplots"""
        t_signal      = np.arange(len(signal)) / ANALOG_RATE
        fig, (ax_raw, ax) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
        fig.suptitle(f'Muskel: {muscle}  |  Linksklick: Onset/Offset setzen  |  Rechtsklick: Phase löschen', fontsize=10)

        """Rohsignal oben"""
        ax_raw.plot(t_signal, signal, color='steelblue', linewidth=0.5)
        ax_raw.set_ylabel('Amplitude')
        ax_raw.set_title('Rohsignal')

        """Wavelet-Energie unten"""
        ax.plot(t_envelope, envelope, color='darkorange', linewidth=1)
        ax.set_xlabel('Zeit [s]')
        ax.set_ylabel('Energie')
        titel = ax.set_title(f'{len(current_phases)} Phasen vorgeschlagen (Bestätigen oder neu setzen)')

        """vorgeschlagene Phasen in beiden Plots einzeichnen"""
        raw_patches = []
        for onset_t, offset_t in current_phases:
            patch     = ax.axvspan(onset_t, offset_t, alpha=0.3, color='green')
            patch_raw = ax_raw.axvspan(onset_t, offset_t, alpha=0.3, color='green')
            phase_patches.append(patch)
            raw_patches.append(patch_raw)

        ax_btn        = plt.axes([0.60, 0.02, 0.2,  0.06])
        ax_reset      = plt.axes([0.15, 0.02, 0.15, 0.06])
        ax_undo       = plt.axes([0.38, 0.02, 0.10, 0.06])
        ax_clear      = plt.axes([0.50, 0.02, 0.08, 0.06])
        button        = Button(ax_btn,   'Bestätigen')
        btn_reset     = Button(ax_reset, 'Zoom Reset')
        btn_undo      = Button(ax_undo,  'Rückgängig')
        btn_clear     = Button(ax_clear, 'Alles löschen')

        def on_click(event):
            if event.inaxes not in [ax, ax_raw]:
                return

            """Rechtsklick: Phase unter Mauszeiger löschen"""
            if event.button == 3:
                t = event.xdata

                getroffen = None
                for idx, (onset_t, offset_t) in enumerate(current_phases):
                    if onset_t <= t <= offset_t:
                        getroffen = idx
                        break

                if getroffen is not None:
                    current_phases.pop(getroffen)
                    phase_patches.pop(getroffen).remove()
                    raw_patches.pop(getroffen).remove()
                    titel.set_text(f'{len(current_phases)} Phasen gesetzt')
                    fig.canvas.draw_idle()
                elif len(current_clicks) > 0:
                    current_clicks.clear()
                    onset_lines.pop().remove()
                    fig.canvas.draw_idle()
                return

            """Linksklick: Onset oder Offset setzen"""
            if event.button == 1:
                t = event.xdata

                if len(current_clicks) == 0:
                    """erster Klick = Onset"""
                    current_clicks.append(t)
                    line = ax.axvline(t, color='green', linewidth=1.5, linestyle='--')
                    onset_lines.append(line)
                    fig.canvas.draw_idle()

                else:
                    """zweiter Klick = Offset"""
                    onset_t  = current_clicks.pop()
                    offset_t = t

                    if onset_t > offset_t:
                        onset_t, offset_t = offset_t, onset_t

                    current_phases.append((onset_t, offset_t))
                    onset_lines.pop().remove()

                    patch     = ax.axvspan(onset_t, offset_t, alpha=0.3, color='green')
                    patch_raw = ax_raw.axvspan(onset_t, offset_t, alpha=0.3, color='green')
                    phase_patches.append(patch)
                    raw_patches.append(patch_raw)
                    titel.set_text(f'{len(current_phases)} Phasen gesetzt')
                    fig.canvas.draw_idle()

        def on_scroll(event):
            if event.inaxes not in [ax, ax_raw]:
                return
            scale     = 0.7 if event.button == 'up' else 1.4
            xlim      = ax.get_xlim()
            xdata     = event.xdata
            new_width = (xlim[1] - xlim[0]) * scale
            ax.set_xlim([xdata - new_width * 0.5, xdata + new_width * 0.5])
            ax_raw.set_xlim([xdata - new_width * 0.5, xdata + new_width * 0.5])
            fig.canvas.draw_idle()

        def reset_zoom(event):
            ax.set_xlim([t_envelope[0], t_envelope[-1]])
            fig.canvas.draw_idle()

        def rueckgaengig(event):
            if len(current_clicks) > 0:
                current_clicks.clear()
                line = onset_lines.pop()
                line.remove()
                titel.set_text(f'{len(current_phases)} Phasen gesetzt')
                fig.canvas.draw_idle()
            elif len(current_phases) > 0:
                current_phases.pop()
                phase_patches.pop().remove()
                raw_patches.pop().remove()
                titel.set_text(f'{len(current_phases)} Phasen gesetzt')
                fig.canvas.draw_idle()

        def alles_loeschen(event):
            current_phases.clear()
            current_clicks.clear()
            for patch in phase_patches:
                patch.remove()
            for patch in raw_patches:
                patch.remove()
            phase_patches.clear()
            raw_patches.clear()
            for line in onset_lines:
                line.remove()
            onset_lines.clear()
            titel.set_text('0 Phasen gesetzt')
            fig.canvas.draw_idle()

        def bestaetigen(event):
            plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        button.on_clicked(bestaetigen)
        btn_reset.on_clicked(reset_zoom)
        btn_undo.on_clicked(rueckgaengig)
        btn_clear.on_clicked(alles_loeschen)

        plt.show()

        """Phasen speichern"""
        labels[muscle] = [[onset_t, offset_t] for onset_t, offset_t in current_phases]
        print(f"Muskel {muscle}: {len(current_phases)} Phasen gespeichert")

    """speichere alle Labels als JSON"""
    result = {file_name: labels}
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"\nLabels gespeichert in: {output_path}")
    return result


if __name__ == "__main__":
    FILE_PATH = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AH0805\AH0805_Isometric_Flexion_Max_Right_2025_05_08.c3d"
    label_file(FILE_PATH)
