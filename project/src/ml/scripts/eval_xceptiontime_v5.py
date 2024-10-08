import torch
import os
import numpy as np
from skorch.helper import SliceDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, precision_score, \
    f1_score
from src.dataset.cropped_dataset import CroppedDataset
from src.ml.utils import xceptiontime_v5
from src.settings import tuab_150hz_dir, cp_xceptiontime_v5

dataset_root = tuab_150hz_dir
file_list = "eval_list"
checkpoint = cp_xceptiontime_v5

device = "cuda" if torch.cuda.is_available() else "cpu"

metrics = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
           "f1": f1_score, "precision": precision_score, "recall": recall_score, "roc_auc": roc_auc_score}


dataset = CroppedDataset(root=dataset_root, file_list="eval_list_first_min", buffer=False)

data = SliceDataset(dataset, idx=0)
targets = dataset.get_class_labels()

data_shape = data[0].shape[0]

class_weights = torch.tensor([1 - np.count_nonzero(targets == 0) / len(targets), 1 - np.count_nonzero(targets == 1) / len(targets)])

score = {key: [] for key in metrics.keys()}

estimator = xceptiontime_v5(device=device, c_in=data_shape, class_weights=class_weights)
estimator.initialize()
estimator.load_params(f_params=f"{checkpoint}params.pt")
pred = estimator.predict(data)
for key, item in metrics.items():
    score[key].append(item(targets, pred))

with open(f"{os.path.basename(__file__).split('.')[0]}.txt", "w") as f:
    print(f"{np.unique(targets, return_counts=True)}", file=f)
    print(f"Data size: {data.shape}", file=f)
    for key, item in score.items():
        print(f"{key}: {np.mean(item):.4f} +/- {np.std(item):.4f}", file=f)
