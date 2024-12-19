import torch
import os
import numpy as np
from skorch.helper import SliceDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, precision_score, \
    f1_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from src.dataset.cropped_dataset import CroppedDataset
from src.ml.utils import xceptiontime_v5
from src.settings import tuab_150hz_dir, cp_xceptiontime_v5, cp_train_xceptiontime_v5

dataset_root = tuab_150hz_dir
file_list = "eval_list"
checkpoint = cp_train_xceptiontime_v5

device = "cuda" if torch.cuda.is_available() else "cpu"
print('====device', device)

metrics = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
           "f1": f1_score, "precision": precision_score, "recall": recall_score, "roc_auc": roc_auc_score}

dataset = CroppedDataset(root=dataset_root, file_list="eval_list_first_min", buffer=False)
data = SliceDataset(dataset, idx=0)
targets = dataset.get_class_labels()

data_shape = data[0].shape[0]

class_weights = torch.tensor([1 - np.count_nonzero(targets == 0) / len(targets), 
                              1 - np.count_nonzero(targets == 1) / len(targets)])
print('===class_weights', class_weights)

# Learning rates and checkpoints
learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
checkpoint_dir = "experiments"  # Folder containing all the checkpoints

# Loop through learning rates and evaluate
for lr in learning_rates:
    lr_folder = os.path.join(checkpoint_dir, f"reduceLRonplateau_{lr}", "checkpoints")
    eval_folder = os.path.join(checkpoint_dir, f"reduceLRonplateau_{lr}", "eval")
    os.makedirs(eval_folder, exist_ok=True)  # Create evaluation folder for each LR
    
    checkpoint_path = os.path.join(lr_folder, "params.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for LR={lr}: {checkpoint_path}")
        continue
    
    print(f"Evaluating for LR={lr} using checkpoint: {checkpoint_path}")
    
    # Initialize the model
    estimator = xceptiontime_v5(device=device, c_in=data_shape, class_weights=class_weights)
    estimator.initialize()
    estimator.load_params(f_params=checkpoint_path)
    print('====done loading params')
    
    # Predict and calculate metrics
    pred = estimator.predict(data)
    pred_prob = estimator.predict_proba(data)  # Get predicted probabilities for AUC-ROC
    print('====pred', pred)
    
    # Compute metrics
    score = {key: [] for key in metrics.keys()}
    for key, item in metrics.items():
        try:
            score[key].append(item(targets, pred))
        except ValueError as e:
            print(f"Error calculating {key}: {e}")
            score[key].append(None)
    
    # Save classification report
    report = classification_report(targets, pred, target_names=["Class 0", "Class 1"])
    report_file = os.path.join(eval_folder, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_file}")
    
    # Compute class-wise AUC-ROC and save plots
    auc_roc_curves_file = os.path.join(eval_folder, "auc_roc_curves.png")
    plt.figure(figsize=(10, 6))
    for i in range(len(np.unique(targets))):  # Iterate over each class
        fpr, tpr, _ = roc_curve(targets == i, pred_prob[:, i])  # One-vs-rest AUC-ROC
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_score:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"AUC-ROC Curves for LR={lr}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(auc_roc_curves_file)
    plt.close()
    print(f"AUC-ROC curves saved to {auc_roc_curves_file}")
    
    # Save metrics to file
    eval_file = os.path.join(eval_folder, "evaluation.txt")
    with open(eval_file, "w") as f:
        print(f"Evaluation results for LR={lr}", file=f)
        print(f"{np.unique(targets, return_counts=True)}", file=f)
        print(f"Data size: {data.shape}", file=f)
        for key, item in score.items():
            if item[0] is not None:
                print(f"{key}: {np.mean(item):.4f} +/- {np.std(item):.4f}", file=f)
            else:
                print(f"{key}: Error calculating metric", file=f)
    print(f"Results for LR={lr} saved to {eval_file}")
