from src.dataset.cropped_dataset import CroppedDataset
from scipy import signal


class FilterDataset(CroppedDataset):
    def __init__(self, srate=1, freqs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.srate = srate
        self.freqs = freqs

    def load(self, idx):
        data = super().load(idx)
        sos = signal.butter(10, self.freqs, "bp", fs=self.srate, output="sos")
        data = signal.sosfilt(sos, data)
        data = data.astype("float32")
        return data
