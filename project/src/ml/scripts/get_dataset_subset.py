import pandas as pd
import random

# Load the dataset
data_path = "/home/zainab/Documents/FYP/Codes/tcav-in-eeg/data/data/datasets/tuab_150hz/train_list.csv"  # Update with your dataset path
data = pd.read_csv(data_path)

# Filter dataset by labels
class_0 = data[data['class'] == 0]
class_1 = data[data['class'] == 1]

# Calculate 10% subset for each class
subset_size = len(data) // 10
subset_size_per_class = subset_size // 2

# Randomly sample from each class
subset_class_0 = class_0.sample(n=subset_size_per_class, random_state=42)
subset_class_1 = class_1.sample(n=subset_size_per_class, random_state=42)

# Combine and shuffle the subsets
subset = pd.concat([subset_class_0, subset_class_1]).sample(frac=1, random_state=42)

# Save the subset to a new file
output_path = "/home/zainab/Documents/FYP/Codes/tcav-in-eeg/data/data/datasets/tuab_150hz/subset_train_list.csv"
subset.to_csv(output_path, index=False)

print(f"Subset saved to {output_path}")
