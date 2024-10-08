from src.dataset.dataset import Dataset


class CroppedDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.overlap = 0.5
        super().__init__(*args, **kwargs)

    def load_from_file(self, idx):
        data = super().load_from_file(idx)
        sample_start = int(self.file_list["sample_idx"][idx] * self.file_list["sample_len"][idx] * self.overlap)
        sample_end = int(sample_start + self.file_list["sample_len"][idx])
        data = data[:, sample_start:sample_end]
        return data
