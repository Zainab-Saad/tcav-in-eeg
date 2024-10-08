import torch.utils.data as data
import numpy as np
import os
import pandas as pd
import copy
from src.dataset.augmentation import mag_warp, time_warp, add_noise


class Dataset(data.Dataset):
    """
    Dataset
    """

    def __init__(self, root=None, flat_features=False, random_n_samples=None, return_target=True, augment=False,
                 file_list="train_list", buffer=False, verbose=False, redefine_classes=None,
                 randomize_per_channel=False, random_state=0):
        self.root = root
        self.flat_features = flat_features
        self.return_target = return_target
        self.augment = augment
        self.randomize_per_channel = randomize_per_channel
        self.verbose = verbose
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.buffer = buffer
        self.buffered_data = []

        file_path = None
        if self.root is not None:
            file_path = os.path.join(self.root, file_list)
        self.file_list = self.load_file_list(file_path)

        if redefine_classes is not None:
            self.redefine_classes(groups=redefine_classes)

        if random_n_samples is not None:
            random_idxs = self.rng.choice(range(len(self.file_list)), size=random_n_samples, replace=False)
            self.file_list = self.file_list.iloc[random_idxs, :]
            self.file_list.reset_index(drop=True, inplace=True)

        # Buffer data
        if buffer:
            self.load_to_buffer()

    def load_file_list(self, file_path):
        if os.path.isfile(file_path + ".csv"):
            file_list = pd.read_csv(file_path + ".csv", sep=",")
        else:
            raise NameError(f"No file list found.")
        return file_list

    def load_to_buffer(self):
        for idx in range(len(self)):
            self.buffered_data.append(self.load_from_file(idx))

    def load_from_file(self, idx):
        if self.verbose:
            if (idx % 100 == 0) or (idx == len(self) - 1):
                print(f"Loading: {idx + 1} / {len(self)}")
        file_path = os.path.join(self.root, self.file_list["path"][idx])
        file_type = os.path.split(file_path)[1].split(sep=".")[-1]
        if file_type == "csv":
            data = np.loadtxt(file_path, delimiter=",", dtype="float32")
        else:
            raise NameError(f"File type {file_type} is not supported")
        return data

    def load(self, idx):
        """
        Load a single timeseries object at given index
        """
        if self.buffered_data:
            data = self.buffered_data[idx]
        else:
            data = self.load_from_file(idx)

        if self.flat_features:
            data = data.flatten()

        if self.augment:
            knots = np.random.randint(low=20, high=120, size=2)
            seeds = np.random.randint(low=0, high=10000, size=3)
            data = mag_warp(data, sigma=0.10, knot=knots[0], random_state=seeds[0],
                            randomize_per_channel=self.randomize_per_channel)
            data = time_warp(data, sigma=0.10, knot=knots[1], random_state=seeds[1],
                             randomize_per_channel=self.randomize_per_channel)
            data = add_noise(data, sigma=0.05, random_state=seeds[2],
                             randomize_per_channel=True)

        return data

    def get_class_labels(self, idx=None):
        """
        Load all class labels or the label for a certain index
        """
        if idx is not None:
            return self.file_list["class"][idx]

        return self.file_list["class"].values

    def copy(self):
        return copy.deepcopy(self)

    def redefine_classes(self, groups=None):
        """
        groups={0: 0, 1: 0, 2: 1}
        Combine original classes 0 and 1 to new class 0 and set original class 2 to new class 1.
        """
        if groups is None:
            return
        # First remove all classes that are not in the specified groups
        all_classes_to_keep = list(groups.keys())
        self.file_list = self.file_list[self.file_list["class"].isin(all_classes_to_keep)]
        # Next rename
        self.file_list.replace({"class": groups}, inplace=True)
        # Reset index
        self.file_list.reset_index(drop=True, inplace=True)

    def get_feature_matrix(self):
        """
        Return all feature matrices
        """
        fv = []
        for idx in range(len(self)):
            if self.return_target:
                fv.append(self[idx][0])
            else:
                fv.append(self[idx])
        fv = np.stack(fv)
        return fv

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.load(idx)
        label = self.get_class_labels(idx)
        if self.return_target:
            return data, label
        else:
            return data
