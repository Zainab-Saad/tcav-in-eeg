from pathlib import Path
from src.dataset.generated_dataset import GeneratedDataset
from src.images.signal_plot import plot_signal
from src.settings import generated_150hz_dir

dataset_root = generated_150hz_dir
Path(f"{dataset_root}/images/").mkdir(parents=True, exist_ok=True)  # Folder for the train subset

freq_bands = {"delta": [[1, 3]],
              "theta": [[4, 7]],
              "alpha": [[8, 13]],
              "beta": [[14, 30]],
              "gamma": [[31, 50]]}

for freq_band_name, freq_band in freq_bands.items():
    dataset = GeneratedDataset(dataset_size=10, timeseries_len=900, n_channels=20, random_seed=0,
                               amplitude=[5, 25], srate=150, freqs=freq_band, randomize_per_channel=False,
                               return_target=False)
    for idx in range(2):
        data = dataset[idx]
        plt_name = f"{dataset_root}/images/{freq_band_name}_{idx:05d}"
        plot_signal(data, all_event_masks=None, srate=150, plt_name=plt_name, legend=False, yticks=False,
                    xtick_step_in_sec=1, fig_width_per_sec=0.6, file_type="png")

