from src.dataset.cropped_dataset import CroppedDataset
from src.settings import tuab_150hz_dir, tueg_150hz_dir, tuev_150hz_dir

tuab_root = tuab_150hz_dir
tueg_root = tueg_150hz_dir
tuev_root = tuev_150hz_dir


dataset = CroppedDataset(root=tuab_root, file_list="train_list")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUAB samples:")
print(class_counts)
file_list.drop_duplicates(subset=["original_file_name"], inplace=True)
class_counts = file_list["class"].value_counts().sort_index()
print("TUAB subjects:")
print(class_counts)

dataset = CroppedDataset(root=tuev_root, file_list="train_list")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUEV samples:")
print(class_counts)

dataset = CroppedDataset(root=tueg_root, file_list="train_list_male")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUEG samples (male):")
print(class_counts)

dataset = CroppedDataset(root=tueg_root, file_list="train_list_female")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUEG samples (female):")
print(class_counts)

dataset = CroppedDataset(root=tueg_root, file_list="train_list_old")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUEG samples (old):")
print(class_counts)

dataset = CroppedDataset(root=tueg_root, file_list="train_list_young")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUEG samples (young):")
print(class_counts)

dataset = CroppedDataset(root=tueg_root, file_list="train_list_random")
file_list = dataset.file_list
class_counts = file_list["class"].value_counts().sort_index()
print("TUEG samples:")
print(class_counts)
