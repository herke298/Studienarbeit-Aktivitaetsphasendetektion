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
    colors      = plt.cm.tab20.colors

    # ----------------------------------------------------------------
    # Interaktives Fenster
    # ----------------------------------------------------------------
    fig, (ax_radio, ax_plot) = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={"width_ratios": [1, 4]})
    fig.patch.set_facecolor("#f0f0f0")
    fig.subplots_adjust(left=0.02, right=0.95)

    ax_plot.set_facecolor("white")
    ax_plot.set_xlabel("[s]", fontsize=10)
    ax_plot.set_ylabel("[µV]", fontsize=10)
    ax_plot.spines[["top", "right"]].set_visible(False)
    ax_plot.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax_plot.axhline(0, color="#aaaaaa", linewidth=0.6)
    ax_plot.axvline(0, color="limegreen", linewidth=1.5)

    signal0 = analogs[emg_indices[0]]
    color0  = colors[0]
    line, = ax_plot.plot(t, signal0, linewidth=0.6, color=color0)
    fill  = ax_plot.fill_between(t, signal0, 0, alpha=0.4, color=color0)
    ax_plot.set_title(emg_labels[0], fontsize=11, color=color0)

    ax_radio.set_facecolor("#f0f0f0")
    radio = RadioButtons(ax_radio, emg_labels, activecolor="steelblue")
    for i, label in enumerate(radio.labels):
        label.set_color(colors[i % len(colors)])
        label.set_fontsize(9)

    def on_select(label):
        idx        = emg_labels.index(label)
        channel_idx = emg_indices[idx]
        color      = colors[idx % len(colors)]
        signal     = analogs[channel_idx]
        line.set_ydata(signal)
        line.set_color(color)
        # Fill neu zeichnen
        for coll in ax_plot.collections:
            coll.remove()
        ax_plot.fill_between(t, signal, 0, alpha=0.4, color=color)
        ax_plot.set_title(label, fontsize=11, color=color)
        ax_plot.relim()
        ax_plot.autoscale_view()
        fig.canvas.draw_idle()

    radio.on_clicked(on_select)

    plt.show()

    return {emg_labels[j]: analogs[idx] for j, idx in enumerate(emg_indices)}

r"""
FILE_PATH1 = r"C:\Studienarbeit\Studienarbeit_Codeuebergabe\Messdaten\AD0805\AD0805_Isometric_Extension_Max_Left_2025_05_08.c3d"
readC3D(FILE_PATH1)
"""