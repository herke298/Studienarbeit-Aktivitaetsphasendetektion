import sys
sys.path.append(r"C:\Studienarbeit\Arbeitsordner\Python Codes")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from emg_utils import compute_rms, compute_doubleThreshold, detect_doublePhases

# Parameter
ANALOG_RATE    = 2000
WINDOW_MS      = 50
window_samples = int(WINDOW_MS / 1000 * ANALOG_RATE)


"""
Interaktiver Plot mit Schiebereglern für T1 und T2
"""
def plot_doubleThreshold_slider(emg_data, t_signal):
    muscles = list(emg_data.keys())
    colors  = plt.cm.tab20.colors

    # RMS für alle Muskeln vorberechnen
    rms_data = {}
    for muscle in muscles:
        rms_data[muscle] = compute_rms(emg_data[muscle], window_samples)
    t_rms = t_signal[:len(rms_data[muscles[0]])]

    # Layout
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#f0f0f0")

    ax_radio  = fig.add_axes([0.01, 0.2, 0.10, 0.7])
    ax_plot   = fig.add_axes([0.14, 0.25, 0.83, 0.68])
    ax_t1     = fig.add_axes([0.14, 0.10, 0.83, 0.04])
    ax_t2     = fig.add_axes([0.14, 0.03, 0.83, 0.04])

    ax_plot.set_facecolor("white")
    ax_plot.spines[["top", "right"]].set_visible(False)
    ax_plot.set_ylabel("[µV]")
    ax_plot.set_xlabel("[s]")

    # Schieberegler
    slider_t1 = Slider(ax_t1, "T1 std", 1.0, 50.0, valinit=5.0, valstep=0.5, color="steelblue")
    slider_t2 = Slider(ax_t2, "T2 std", 1.0, 50.0, valinit=2.0, valstep=0.5, color="darkorange")

    # Initiale Daten
    muscle0 = muscles[0]
    color0  = colors[0]
    t1, t2  = compute_doubleThreshold(rms_data[muscle0], slider_t1.val, slider_t2.val)

    line_raw,  = ax_plot.plot(t_signal, emg_data[muscle0], linewidth=0.5, color=color0, alpha=0.8)
    line_t1    = ax_plot.axhline(t1, color="steelblue",  linewidth=1.0, linestyle="--", label="T1")
    line_t2    = ax_plot.axhline(t2, color="darkorange", linewidth=1.0, linestyle="--", label="T2")
    ax_plot.legend(fontsize=9, loc="upper right")

    phase_lines = []
    phases = detect_doublePhases(rms_data[muscle0], t1, t2)
    for onset_idx, offset_idx in phases:
        l1 = ax_plot.axvline(t_rms[onset_idx],  color="green",  linewidth=1.2)
        l2 = ax_plot.axvline(t_rms[offset_idx], color="purple", linewidth=1.2)
        phase_lines.extend([l1, l2])

    fig.suptitle(f"{muscle0}  —  {len(phases)} Phase(n)  |  T1={slider_t1.val:.1f}  T2={slider_t2.val:.1f}", fontsize=11, x=0.58)

    # Radio-Buttons
    ax_radio.set_facecolor("#f0f0f0")
    radio = RadioButtons(ax_radio, muscles, activecolor="steelblue")
    for i, label in enumerate(radio.labels):
        label.set_color(colors[i % len(colors)])
        label.set_fontsize(9)

    def update(val=None):
        idx    = muscles.index(current_muscle[0])
        color  = colors[idx % len(colors)]
        muscle = current_muscle[0]
        t1, t2 = compute_doubleThreshold(rms_data[muscle], slider_t1.val, slider_t2.val)

        line_raw.set_ydata(emg_data[muscle])
        line_raw.set_color(color)
        line_t1.set_ydata([t1, t1])
        line_t2.set_ydata([t2, t2])

        for l in phase_lines:
            l.remove()
        phase_lines.clear()

        phases = detect_doublePhases(rms_data[muscle], t1, t2)
        for onset_idx, offset_idx in phases:
            l1 = ax_plot.axvline(t_rms[onset_idx],  color="green",  linewidth=1.2)
            l2 = ax_plot.axvline(t_rms[offset_idx], color="purple", linewidth=1.2)
            phase_lines.extend([l1, l2])

        fig.suptitle(f"{muscle}  —  {len(phases)} Phase(n)  |  T1={slider_t1.val:.1f}  T2={slider_t2.val:.1f}", fontsize=11, x=0.58)

        ax_plot.relim()
        ax_plot.autoscale_view()
        fig.canvas.draw_idle()

    current_muscle = [muscle0]

    def on_select(label):
        current_muscle[0] = label
        update()

    slider_t1.on_changed(update)
    slider_t2.on_changed(update)
    radio.on_clicked(on_select)
    plt.show()
