from skorch.helper import SliceDataset
import os.path
import mlflow.exceptions
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetClassifier
from skorch.callbacks import MlflowLogger, EpochScoring, LRScheduler, Checkpoint, EarlyStopping
from torch import nn
from torch import optim
import torch
from tsai.models.XceptionTimePlus import XceptionTimePlus
from mlflow.tracking import MlflowClient
from src.dataset.cropped_dataset import CroppedDataset
from src.settings import tuab_150hz_dir

dataset_root = tuab_150hz_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

client = MlflowClient()
try:
    experiment = client.create_experiment("TUH")
except mlflow.exceptions.MlflowException:
    experiment = client.get_experiment_by_name("TUH")
tags = {"script_name": os.path.basename(__file__)}
experiment_id = experiment.experiment_id if hasattr(experiment, 'experiment_id') else experiment
run = client.create_run(experiment_id, run_name=os.path.basename(__file__), tags=tags) # then i did this and this works even when experiment is created
# run = client.create_run(experiment.experiment_id, run_name=os.path.basename(__file__), tags=tags)  # this was original
# run = client.create_run(experiment, run_name=os.path.basename(__file__), tags=tags) # first i did this but this gave error if experiment was already created

def get_estimator(data_shape, class_weights, valid_sampler):
    net = NeuralNetClassifier(module=XceptionTimePlus, module__c_in=data_shape, module__c_out=2,
                              module__nf=12, module__act=nn.LeakyReLU,
                              max_epochs=100,
                              iterator_train__shuffle=True,
                              criterion=nn.CrossEntropyLoss, criterion__weight=class_weights, optimizer=optim.AdamW,
                              train_split=valid_sampler,
                              batch_size=32, lr=0.001, callbacks=[EpochScoring("accuracy",
                                                                               lower_is_better=False,
                                                                               name="train_acc",
                                                                               on_train=True),
                                                                  LRScheduler(policy=optim.lr_scheduler.OneCycleLR,
                                                                              monitor="valid_loss", max_lr=0.0005,
                                                                              epochs=1,
                                                                              steps_per_epoch=100,
                                                                              div_factor=10,
                                                                              final_div_factor=1000,
                                                                              step_every="epoch"),
                                                                  EarlyStopping(patience=25),
                                                                  Checkpoint(dirname=f"cp_{os.path.basename(__file__).split('.')[0]}"),
                                                                  MlflowLogger(run, client)
                                                                  ],
                              device=device)
    return net


class GroupedSampler:
    def __init__(self, sampler, groups):
        self.sampler = sampler
        self.groups = groups

    def __call__(self, dataset, y, **fit_params):
        train_idx, test_idx = next(self.sampler.split(np.ones_like(sample_ids), np.ones_like(sample_ids),
                                                      self.groups))
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, test_idx)
        return train_dataset, val_dataset


dataset = CroppedDataset(root=dataset_root, file_list="train_list", buffer=False)
# sample_ids = [file["original_file_name"][:8] for file in dataset.file_list]
sample_ids = dataset.file_list["original_file_name"].str[:8].tolist()
# sample_ids = sample_ids[20045:20331]
sample_ids = LabelEncoder().fit_transform(sample_ids)
unique_ids = np.unique(sample_ids)
valid_sampler = GroupedSampler(GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.1), sample_ids)

print('===starting pipeline')
# data = SliceDataset(dataset, idx=0)
data = dataset.get_feature_matrix()
data = np.asarray(data)
data = data.astype("float32")

print('====data', data)
targets = dataset.get_class_labels()

targets = np.asarray(targets)
targets = targets.astype("int64")
# targets = targets[20045:20331]
print('====targets', targets)

data_shape = data[0].shape[0]
print('=====data_shape', data_shape)
class_weights = torch.tensor([1 - np.count_nonzero(targets == 0) / len(targets), 1 - np.count_nonzero(targets == 1) / len(targets)])
print('=====class_weights', class_weights)
estimator = get_estimator(data_shape, class_weights, valid_sampler)

estimator.fit(data, targets)
