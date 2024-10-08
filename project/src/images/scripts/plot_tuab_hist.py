import matplotlib.pyplot as plt
from src.dataset.cropped_dataset import CroppedDataset
from src.settings import tuab_150hz_dir

dataset_root = tuab_150hz_dir
save_dir = "/home/alex/tcav_in_biosignals/data/images/"
file_list = "train_list"
plt_name = "tuab_hist"
file_type = "png"

dataset = CroppedDataset(root=dataset_root, file_list=file_list)
df = dataset.file_list
# Get individual subjects
df = df.groupby("original_file_name").first()
df.reset_index(drop=True, inplace=True)
# Pandas does not accept "class" in queries
df.rename(columns={"class": "label"}, inplace=True)

gender_count = df["gender"].value_counts(normalize=True)

# Binning parameters
bin_width = 10
max_age = 100
min_age = 0
age_bins = list(range(min_age, max_age + bin_width, bin_width))

# Create a figure with two subplots
fig, axs = plt.subplots(nrows=1, ncols=2, sharey="row", figsize=(10, 6))

# Females
subset = df.query("gender=='F'")
subset_normal = df.query("gender=='F' and label==0")
subset_abnormal = df.query("gender=='F' and label==1")

axs[0].hist([subset_normal["age"], subset_abnormal["age"]], bins=age_bins,
            label=["Normal", "Abnormal"], stacked=True, orientation="horizontal")
axs[0].set_ylabel("Age (years)")
axs[0].set_yticklabels([])
axs[0].set_xlabel("Count")
axs[0].set_title(f"Females ({gender_count['F']:.2%})")
label_count = subset["label"].value_counts(normalize=True)
axs[0].legend([f"Normal ({label_count[0]:.2%})",
               f"Abnormal ({label_count[1]:.2%})"])

axs[0].invert_xaxis()
axs[0].yaxis.tick_right()

# Males
subset = df.query("gender=='M'")
subset_normal = df.query("gender=='M' and label==0")
subset_abnormal = df.query("gender=='M' and label==1")

axs[1].hist([subset_normal["age"], subset_abnormal["age"]], bins=age_bins,
            label=["Normal", "Abnormal"], stacked=True, orientation="horizontal")
# axs[1].set_ylabel("Age (years)")
axs[1].set_yticklabels([])
axs[1].set_xlabel("Count")
axs[1].set_title(f"Males ({gender_count['M']:.2%})")
label_count = subset["label"].value_counts(normalize=True)
axs[1].legend([f"Normal ({label_count[0]:.2%})",
               f"Abnormal ({label_count[1]:.2%})"])

y_labels = age_bins[::2]
axs[1].set_yticks(y_labels)
for y_label in y_labels:
    y_coord = y_label
    axs[1].annotate(y_label, (0.51, y_coord), xycoords=("figure fraction", "data"), ha="center", va="center")

# Adjust spacing between subplots
plt.tight_layout()

xlim_min = min([axs[0].get_xlim()[1], axs[0].get_xlim()[0]])
xlim_max = max([axs[0].get_xlim()[0], axs[1].get_xlim()[1]])
axs[0].set_xlim([xlim_max, xlim_min])
axs[1].set_xlim([xlim_min, xlim_max])

plt_name = save_dir + plt_name
plt.savefig(f"{plt_name}.{file_type}")
print(f"Plotting {plt_name}")
plt.close()