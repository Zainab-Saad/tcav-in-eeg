from pathlib import Path
from src.dataset.event_dataset import EventDataset
from src.images.signal_plot import get_all_event_masks, plot_signal
from src.settings import tuev_150hz_dir

dataset_root = tuev_150hz_dir
dataset_type = "train"


dataset = EventDataset(root=dataset_root, zero_background=False)

Path(f"{dataset_root}/images/").mkdir(parents=True, exist_ok=True)  # Folder for the train subset

for idx in range(100): # 200
    data, event = dataset[idx]
    annot = dataset.file_list["annot"][idx]
    event_blocks = dataset.file_list["event_blocks"][idx]
    srate = dataset.file_list["srate"][idx]
    plt_name = f"{dataset_root}/images/{event:02d}_{idx:05d}"
    all_event_masks = get_all_event_masks(data, annot)
    plot_signal(data, all_event_masks, srate, plt_name, legend=False, yticks=True, xtick_step_in_sec=1, file_type="png")
