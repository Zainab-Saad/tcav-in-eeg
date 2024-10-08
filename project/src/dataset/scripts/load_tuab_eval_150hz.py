import os
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from src.dataset.utils import load_eeg_file, parse_age_and_gender_from_edf_header
from src.settings import tuab_dir, tuab_150hz_dir

dataset_root = tuab_dir
montage = True
srate = 150
# All len values in seconds
sample_len = 60
sample_overlap = 30
max_len = 480
save_path = tuab_150hz_dir
subset = "eval"
verbose = True

Path(save_path + subset + "/").mkdir(parents=True, exist_ok=True)  # Folder for the subset

print(f"Searching for: {dataset_root}**/*.edf")
file_count = 0
file_list = []
st_time = time.time()  # Save starting time
iterator = sorted(glob.glob(dataset_root + "**/*.edf", recursive=True))
for idx, file_path in enumerate(iterator):
    relative_file_path = file_path.replace(dataset_root, "")
    if idx % 100 == 0 or idx == len(iterator) - 1:  # Logging
        time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))  # Time formatted
        if verbose:
            print("Processing time {0}, Step {1} / {2}".format(time_str, idx + 1, len(iterator)))

    if subset not in relative_file_path:
        continue

    if "abnormal" in relative_file_path:
        label = 1
    else:
        label = 0

    channel_data, orig_srate, ch_names, reference = load_eeg_file(file_path, max_len=max_len)
    if channel_data is None:
        print(f"Reading error in file: {relative_file_path}")
        continue

    # Resample
    num = int(len(channel_data[0]) * (srate / orig_srate))
    channel_data = np.apply_along_axis(scipy.signal.resample, 1, channel_data, num)

    # Clipping
    max_val = 800
    channel_data = np.clip(channel_data, max_val * -1, max_val)

    # Cut down to multiple windows with overlap
    n_fits = channel_data.shape[1] // (sample_overlap * srate) - 1
    if n_fits == 0:
        print(f"Record length < sample length: {channel_data.shape[1]} < {sample_len * srate}, file: {relative_file_path}")
        continue
    new_len = (n_fits + 1) * sample_overlap * srate
    if new_len < max_len * srate:
        print(f"Record length < max. defined length: {new_len} < {max_len * srate}, file: {relative_file_path}")
        channel_data = channel_data[:, :new_len]

    # Get original file name
    original_file_name = file_path.split("/")[-1].split(".")[0]

    age, gender = parse_age_and_gender_from_edf_header(file_path)

    file_count += 1
    data_save_name = subset + f"/{file_count:05d}" + ".csv"  # Name of file to be saved

    for sample_idx in range(n_fits):
        file_list.append({"path": data_save_name, "nsamples": n_fits, "class": label,
                          "age": age, "gender": gender,
                          "sample_idx": sample_idx,
                          "srate": srate,
                          "ch_names": ch_names,
                          'sample_len': sample_len * srate, "original_file_name": original_file_name})
    np.savetxt(os.path.join(save_path, data_save_name), channel_data, delimiter=",")  # Save eeg file

file_list = pd.DataFrame(file_list)
file_list.to_csv(os.path.join(save_path, f"{subset}_list.csv"), sep=",", index=False)


