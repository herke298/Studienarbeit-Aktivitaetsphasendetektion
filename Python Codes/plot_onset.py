import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons


"""
Interaktiver Plot für Onset-Detektion.
Erwartet:
  - emg_data   : Dict {muskelname: rohsignal}
  - phases     : Dict {muskelname: [(onset_idx, offset_idx), ...]}
  - t_signal   : Zeitachse
  - title      : Titel des Plots (z.B. "Simple Threshold")
"""
def plot_onset(emg_data, phases, t_signal, title):
    muscles = list(emg_data.keys())
    colors  = plt.cm.tab20.colors

    # Layout
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#f0f0f0")

    ax_radio = fig.add_axes([0.01, 0.1, 0.12, 0.8])
    ax_raw   = fig.add_axes([0.16, 0.1, 0.81, 0.8])

    ax_raw.set_facecolor("white")
    ax_raw.spines[["top", "right"]].set_visible(False)
    ax_raw.set_ylabel("[µV]")
    ax_raw.set_xlabel("[s]")

    # Initiale Daten
    muscle0 = muscles[0]
    color0  = colors[0]

    line_raw, = ax_raw.plot(t_signal, emg_data[muscle0], linewidth=0.5, color=color0, alpha=0.8)

    phase_lines = []
    for onset_idx, offset_idx in phases[muscle0]:
        l1 = ax_raw.axvline(t_signal[onset_idx],  color="green",  linewidth=1.2)
        l2 = ax_raw.axvline(t_signal[offset_idx], color="purple", linewidth=1.2)
        phase_lines.extend([l1, l2])

    fig.suptitle(f"{title}  |  {muscle0}  —  {len(phases[muscle0])} Phase(n)", fontsize=12, x=0.58)

    # Radio-Buttons
    ax_radio.set_facecolor("#f0f0f0")
    radio = RadioButtons(ax_radio, muscles, activecolor="steelblue")
    for i, label in enumerate(radio.labels):
        label.set_color(colors[i % len(colors)])
        label.set_fontsize(9)

    def on_select(label):
        idx   = muscles.index(label)
        color = colors[idx % len(colors)]

        line_raw.set_ydata(emg_data[label])
        line_raw.set_color(color)

        for l in phase_lines:
            l.remove()
        phase_lines.clear()

        for onset_idx, offset_idx in phases[label]:
            l1 = ax_raw.axvline(t_signal[onset_idx],  color="green",  linewidth=1.2)
            l2 = ax_raw.axvline(t_signal[offset_idx], color="purple", linewidth=1.2)
            phase_lines.extend([l1, l2])

        fig.suptitle(f"{title}  |  {label}  —  {len(phases[label])} Phase(n)", fontsize=12, x=0.58)

        ax_raw.relim()
        ax_raw.autoscale_view()
        fig.canvas.draw_idle()

    radio.on_clicked(on_select)
    plt.show()
