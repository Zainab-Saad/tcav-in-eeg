from ast import literal_eval
from pathlib import Path
import numpy as np
from src.dataset.dataset import Dataset
from src.images.signal_plot import get_all_event_masks, plot_signal
from src.settings import tuev_150hz_dir

dataset_root = tuev_150hz_dir
dataset_type = "train"


dataset = Dataset(root=dataset_root)

Path(f"{dataset_root}/images/records").mkdir(parents=True, exist_ok=True)  # Folder for the train subset

file_list = dataset.file_list.copy()
file_list = file_list.loc[file_list["sample_idx"] == 0, :]

file_list["annot"] = file_list["annot"].apply(lambda s: np.array(literal_eval(s)))
file_list["event_blocks"] = file_list["event_blocks"].apply(lambda s: np.array(literal_eval(s)))

for counter, idx in enumerate(file_list.index):
    data, event = dataset[idx]
    annot = file_list["annot"][idx]
    event_blocks = file_list["event_blocks"][idx]
    srate = file_list["srate"][idx]

    annot[:, 1:3] *= srate
    annot = annot.astype(int)
    event_blocks[:, 0:2] *= srate
    event_blocks = event_blocks.astype(int)

    plt_name = f"{dataset_root}/images/records/{idx:05d}"
    all_event_masks = get_all_event_masks(data, annot)
    plot_signal(data, all_event_masks, srate, plt_name, legend=True, yticks=True, fig_width_per_sec=0.2,
                event_blocks=event_blocks)

    if counter == 200:
        break
