from src.dataset.augmentation import mag_warp, add_noise, time_warp
from src.dataset.dataset import Dataset
import numpy as np
import pandas as pd


class GeneratedDataset(Dataset):
    def __init__(self, dataset_size=None, timeseries_len=None, n_channels=1, random_seed=0,
                 amplitude=1, srate=1, freqs=[[1]], *args, **kwargs):
        self.dataset_size = dataset_size
        self.file_list_generator = np.random.default_rng(random_seed)
        super().__init__(*args, **kwargs)
        self.timeseries_len = timeseries_len
        self.n_channels = n_channels
        self.amplitude = amplitude
        self.srate = srate
        self.freqs = freqs

    def load_file_list(self, file_path):
        # Generate list of random integers that will represent random seeds per timeseries
        random_list = self.file_list_generator.integers(low=0, high=10000, size=self.dataset_size)
        file_list = []
        for idx, random_element in enumerate(random_list):
            file_list.append({"path": f"{idx}.random", "class": 0, "random_seed": random_element})
        return pd.DataFrame(file_list)

    def load_from_file(self, idx):
        return None

    def load(self, idx):
        data_generator = np.random.default_rng(self.file_list["random_seed"][idx])
        data = np.zeros([self.n_channels, self.timeseries_len])
        for idx, freq in enumerate(self.freqs):
            if isinstance(freq, list):
                freq = data_generator.uniform(freq[0], freq[1])
            start_phase = np.linspace(0, freq, int(self.srate * freq))
            start_phase = data_generator.choice(start_phase, size=1)[0]
            t = np.linspace(0, int(self.timeseries_len / self.srate), self.timeseries_len)
            t += start_phase
            freq_data = np.sin(2 * np.pi * freq * t)
            for ch_idx in range(self.n_channels):
                scale = self.amplitude
                if isinstance(self.amplitude, list):
                    scale = data_generator.uniform(self.amplitude[0], self.amplitude[1])
                data[ch_idx, :] += freq_data * scale

        if self.flat_features:
            data = data.flatten()

        knots = data_generator.integers(low=20, high=120, size=2)
        data = mag_warp(data, sigma=0.10, knot=knots[0], random_state=self.file_list["random_seed"][idx],
                        randomize_per_channel=self.randomize_per_channel)
        data = time_warp(data, sigma=0.10, knot=knots[1], random_state=self.file_list["random_seed"][idx],
                         randomize_per_channel=self.randomize_per_channel)
        data = add_noise(data, sigma=0.05, random_state=self.file_list["random_seed"][idx],
                         randomize_per_channel=True)

        return data
