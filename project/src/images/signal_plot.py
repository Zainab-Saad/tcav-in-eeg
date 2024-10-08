import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ch_names = \
            ["FP1-F7",
            "F7-T3",
            "T3-T5",
            "T5-O1",
            "FP2-F8",
            "F8-T4",
            "T4-T6",
            "T6-O2",
            "T3-C3",
            "C3-CZ",
            "CZ-C4",
            "C4-T4",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2"]


def get_event_mask(data, annot, event=None):
    # Create a mask with False values
    event_mask = np.full(np.array(data.shape), False)
    if event is not None:
        annot = annot[annot[:, 3] == event, :]  # filter event
    for row in annot:
        event_mask[int(row[0]), int(row[1]):int(row[2]) + 1] = True  # Set event indexes to true
    return event_mask


def get_all_event_masks(data, annot):
    """
    Return dict in the format {event_id: event_mask}
    """
    all_event_masks = {}
    for event in np.unique(annot[:, 3]):
        all_event_masks[event] = get_event_mask(data, annot, event=event)
    return all_event_masks


def mask_to_multi_masks(mask):
    multi_masks = []
    # group_idx = -1
    group_mode = False
    # new_mask = np.full(np.array(mask.shape, False))
    for current_idx, element in enumerate(mask):
        if group_mode == False:
            if element == True:
                group_mode = True
                new_mask = np.full(np.array(mask.shape), False)
                new_mask[current_idx] = True
                # group_idx += 1
        if group_mode == True:
            if element == False:
                group_mode = False
                multi_masks.append(new_mask)
            if element == True:
                new_mask[current_idx] = True
    if group_mode == True:
        multi_masks.append(new_mask)
    return multi_masks


def plot_signal(signal_data, all_event_masks=None, srate=150, plt_name="plt", legend=True, yticks=True, fig_width_per_sec=0.6,
                xtick_step_in_sec=5, event_blocks=None, file_type="png"):
    n_channels = signal_data.shape[0]
    times = np.arange(signal_data.shape[1])
    fig_width = (signal_data.shape[1] / srate) * fig_width_per_sec

    offset = 100  # distance between channels
    ticklocs = []
    for idx in range(n_channels):
        signal_data[idx, :] = signal_data[idx, :] + idx * offset
        ticklocs.append(idx * offset)

    fig = plt.figure(figsize=(fig_width, 4))
    ax = plt.subplot()

    # signal_data = np.swapaxes(signal_data, 0, 1)
    for row in signal_data:
        plt.plot(times, row, color="lightskyblue", linewidth=0.5)

    if all_event_masks is not None:
        event_colors = ["purple", "green", "orange", "red", "blue", "yellow"]
        for event_class, event_mask in all_event_masks.items():
            for row, next_event_mask in zip(signal_data, event_mask):
                multi_masks = mask_to_multi_masks(next_event_mask)
                for mask in multi_masks:
                    plt.plot(times[mask], row[mask], color=event_colors[int(event_class)-1], linewidth=0.5)

    if yticks:
        ax.set_yticks(ticklocs, labels=ch_names)
    else:
        ax.set_yticks([])

    ax.set_xticks(np.arange(0, signal_data.shape[1] + 1, srate * xtick_step_in_sec),
                  labels=np.arange(0, (signal_data.shape[1] // srate) + 1, xtick_step_in_sec))

    ax.set_ylim(-100, n_channels * 100 + 100)
    ax.set_xlim(0, signal_data.shape[1])

    ax.set_xlabel("Time (s)")

    ax.grid(axis="y", color="gray", alpha=0.2, linestyle=":")
    ax.grid(axis="x", color="gray", alpha=0.2, linestyle=":")

    if event_blocks is not None:
        for event_block in event_blocks:
            plt.vlines(event_block[:2], -100, n_channels * offset + 100, color=event_colors[event_block[2] - 1])

    if legend:
        legend_elements = [Line2D([0], [0], color=color, lw=2) for color in event_colors]
        ax.legend(legend_elements, ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"], loc="upper right")

    plt.tight_layout()

    # plt.show()
    plt.savefig(f"{plt_name}.{file_type}")
    print(f"Plotting {plt_name}")

    plt.close()