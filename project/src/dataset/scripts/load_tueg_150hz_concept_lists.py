import pandas as pd
import numpy as np
from src.dataset.dataset import Dataset
from src.settings import tuab_150hz_dir, tueg_150hz_dir

tuab_root = tuab_150hz_dir
dataset_root = tueg_150hz_dir

tuab_train = Dataset(root=tuab_root, file_list="train_list")
tuab_eval = Dataset(root=tuab_root, file_list="eval_list")

df_tuab_train = tuab_train.file_list
df_tuab_eval = tuab_eval.file_list
df_tuab = pd.concat([df_tuab_train, df_tuab_eval])
# Reduce to subjects
df_tuab["subject_id"] = df_tuab["original_file_name"].apply(lambda x: x[:8])
df_tuab = df_tuab.groupby("subject_id").first()


dataset = Dataset(root=dataset_root, file_list="train_list")
df_samples = dataset.file_list
df_samples["subject_id"] = df_samples["original_file_name"].apply(lambda x: x[:8])
df_subjects = df_samples.groupby("subject_id").first()

ages = np.array(["mid" for _ in range(len(df_subjects))])
ages[df_subjects["age"] >= 60] = "old"
ages[df_subjects["age"] <= 30] = "young"
df_subjects["age_cat"] = ages

mask = []
for idx in df_subjects.index:
    if idx in df_tuab.index.tolist():
        mask.append(False)
    else:
        mask.append(True)
df_subjects = df_subjects.loc[mask, :]

df_male = df_subjects.loc[df_subjects["gender"] == "M", :]
df_male = df_male[:100]
df_female = df_subjects.loc[df_subjects["gender"] == "F", :]
df_female = df_female[:100]

df_subjects.drop(index=df_male.index, inplace=True)
df_subjects.drop(index=df_female.index, inplace=True)

df_old = df_subjects.loc[df_subjects["age_cat"] == "old", :]
df_old = df_old[:100]
df_young = df_subjects.loc[df_subjects["age_cat"] == "you", :]
df_young = df_young[:100]

df_subjects.drop(index=df_old.index, inplace=True)
df_subjects.drop(index=df_young.index, inplace=True)

df_male.to_csv(dataset_root + "train_list_male.csv", sep=",")
df_female.to_csv(dataset_root + "train_list_female.csv", sep=",")
df_old.to_csv(dataset_root + "train_list_old.csv", sep=",")
df_young.to_csv(dataset_root + "train_list_young.csv", sep=",")
df_subjects.to_csv(dataset_root + "train_list_random.csv", sep=",")
print("done")