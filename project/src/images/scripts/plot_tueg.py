from pathlib import Path
from src.dataset.cropped_dataset import CroppedDataset
from src.images.signal_plot import plot_signal
from src.settings import tueg_150hz_dir

dataset_root = tueg_150hz_dir
dataset_type = "train"


dataset = CroppedDataset(root=dataset_root, file_list="train_list", buffer=False)

Path(f"{dataset_root}/images/").mkdir(parents=True, exist_ok=True)  # Folder for the train subset

for idx in range(5): # 100
    data, label = dataset[idx]
    srate = dataset.file_list["srate"][idx]
    plt_name = f"{dataset_root}/images/{dataset_type}_{idx:05d}"
    plot_signal(data, all_event_masks=None, srate=150, plt_name=plt_name, legend=False, yticks=False,
                xtick_step_in_sec=10, fig_width_per_sec=0.2, file_type="png")
