import os
from src.dataset.dataset import Dataset
from src.settings import tuab_150hz_dir

dataset_root = tuab_150hz_dir
file_list = "eval_list"
new_file_list = "eval_list_first_min"

dataset = Dataset(root=dataset_root, file_list=file_list, buffer=False)

df = dataset.file_list
mask = df["sample_idx"] == 0
df = df.loc[mask, :]

df.to_csv(os.path.join(dataset_root, new_file_list + ".csv"), sep=",", index=False)
