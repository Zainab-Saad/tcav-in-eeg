import pickle
from pathlib import Path

from src.settings import tcav_dir
from src.tcav.plots import plot_tcav_scores

tcav_path = tcav_dir
img_root = tcav_dir + "images/"
img_subdir = "tcav_scores/"
img_dir = img_root + img_subdir

Path(img_dir).mkdir(parents=True, exist_ok=True)  # Folder for the subset

experimental_names = [
    ("random_delta", "random_eeg"),
    ("random_theta", "random_eeg"),
    ("random_alpha", "random_eeg"),
    ("random_beta", "random_eeg"),
    ("random_gamma", "random_eeg"),
]
input_name = "abnormal_test"

all_experimental_sets = []
all_tcav_scores = []
all_stats = []
for idx, exp in enumerate(experimental_names):
    print(f"Experiment {idx+1} / {len(experimental_names)}")
    exp_name = f"{input_name}_{'_'.join(exp)}"
    exp_path = f"{tcav_path}/{exp_name}.pkl"
    with open(exp_path, "rb") as f:
        res = pickle.load(f)
    experimental_sets = res["experimental_sets"]
    tcav_scores = res["tcav_scores"]
    stats = res["stats"]

    all_experimental_sets.append(experimental_sets)
    all_tcav_scores.append(tcav_scores)
    all_stats.append(stats)

exp_name = f"{input_name}_{'_'.join(experimental_names[0])}"
img_path = img_dir + exp_name
plot_tcav_scores(all_experimental_sets, all_tcav_scores, only_significant=True, with_error=False,
                 plt_name=img_path, file_type="png")
