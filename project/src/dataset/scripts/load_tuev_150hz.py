import os
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from src.dataset.utils import load_eeg_file, parse_age_and_gender_from_edf_header, load_annotation, get_event_samples
from src.settings import tuev_dir, tuev_150hz_dir

dataset_root = tuev_dir
montage = True
srate = 150
save_path = tuev_150hz_dir
subset = "train"
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

    channel_data, orig_srate, ch_names, reference = load_eeg_file(file_path)
    if channel_data is None:
        print(f"Reading error in file: {relative_file_path}")
        continue

    # Resample
    num = int(len(channel_data[0]) * (srate / orig_srate))
    channel_data = np.apply_along_axis(scipy.signal.resample, 1, channel_data, num)

    # Clipping
    max_val = 800
    channel_data = np.clip(channel_data, max_val * -1, max_val)

    # Cut down to multiple samples accoridng to the events
    df = load_annotation(file_path)
    annot = df.values
    samples = get_event_samples(annot)
    if samples is None:
        print(f"No events in file: {relative_file_path}")
        continue

    min_start = np.min(samples[:, 0])
    max_end = np.max(samples[:, 1])
    channel_data = channel_data[:, int(min_start * srate):int(max_end * srate)]
    annot[:, 1:3] -= min_start
    samples[:, :2] -= min_start

    annot = annot.tolist()
    sample_len_list = [int((sample[1] - sample[0]) * srate) for sample in samples]
    sample_start_list = [int(sample[0] * srate) for sample in samples]
    label_list = samples.astype(int)[:, 2].tolist()
    nsamples = samples.shape[0]

    # Get original file name
    original_file_name = file_path.split("/")[-1].split(".")[0]

    age, gender = parse_age_and_gender_from_edf_header(file_path)

    file_count += 1
    data_save_name = subset + f"/{file_count:05d}" + ".csv"  # Name of file to be saved

    for sample_idx, (sample_len, sample_start, label) in enumerate(zip(sample_len_list, sample_start_list, label_list)):
        file_list.append({"path": data_save_name, "nsamples": nsamples, "class": label,
                          "age": age, "gender": gender,
                          "sample_start": sample_start,
                          "sample_idx": sample_idx,
                          "srate": srate,
                          "ch_names": ch_names,
                          "sample_len": sample_len, "original_file_name": original_file_name,
                          "annot": annot, "event_blocks": samples.tolist()})
    np.savetxt(os.path.join(save_path, data_save_name), channel_data, delimiter=",")  # Save eeg file

file_list = pd.DataFrame(file_list)
file_list.to_csv(os.path.join(save_path, f"{subset}_list.csv"), sep=",", index=False)


