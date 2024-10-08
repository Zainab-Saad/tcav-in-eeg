from ast import literal_eval
from src.dataset.dataset import Dataset
import numpy as np


class EventDataset(Dataset):
    def __init__(self, zero_background=False, *args, **kwargs):
        self.zero_background = zero_background
        super().__init__(*args, **kwargs)
        self.file_list["annot"] = self.file_list["annot"].apply(lambda s: np.array(literal_eval(s)))
        self.file_list["event_blocks"] = self.file_list["event_blocks"].apply(lambda s: np.array(literal_eval(s)))

    def filter_annot(self, idx):
        annot = self.file_list["annot"][idx]
        annot[:, 1:3] = (annot[:, 1:3] * self.file_list["srate"][idx])
        annot = annot.astype(int)
        sample_end = self.file_list["sample_start"][idx] + self.file_list["sample_len"][idx]
        annot[:, 2] = np.where(annot[:, 2] > sample_end, sample_end, annot[:, 2])
        annot[:, 1] = np.where(annot[:, 1] < self.file_list["sample_start"][idx], self.file_list["sample_start"][idx], annot[:, 1])
        annot[:, 1:3] -= self.file_list["sample_start"][idx]
        mask = (annot[:, 1] < self.file_list["sample_len"][idx]) & (annot[:, 2] > 0)
        annot = annot[mask]
        self.file_list.at[idx, "annot"] = annot

    def load_from_file(self, idx):
        data = super().load_from_file(idx)
        self.filter_annot(idx)
        sample_start = self.file_list["sample_start"][idx]
        sample_end = int(sample_start + self.file_list["sample_len"][idx])
        data = data[:, sample_start:sample_end]
        if self.zero_background:
            data_new = np.zeros_like(data)
            annot = np.array(self.file_list["annot"][idx])
            annot = annot[annot[:, 3] == self.file_list["class"][idx], :]
            for row in annot:
                data_new[row[0], row[1]:row[2]] = data[row[0], row[1]:row[2]]
            data = np.array(data_new, dtype="float32")
        return data
