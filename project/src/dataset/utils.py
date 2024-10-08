import json
import re
import numpy as np
import pandas as pd
import pyedflib

montage_20 = ["EEG FP1-REF -- EEG F7-REF",
              "EEG F7-REF -- EEG T3-REF",
              "EEG T3-REF -- EEG T5-REF",
              "EEG T5-REF -- EEG O1-REF",
              "EEG FP2-REF -- EEG F8-REF",
              "EEG F8-REF -- EEG T4-REF",
              "EEG T4-REF -- EEG T6-REF",
              "EEG T6-REF -- EEG O2-REF",
              "EEG T3-REF -- EEG C3-REF",
              "EEG C3-REF -- EEG CZ-REF",
              "EEG CZ-REF -- EEG C4-REF",
              "EEG C4-REF -- EEG T4-REF",
              "EEG FP1-REF -- EEG F3-REF",
              "EEG F3-REF -- EEG C3-REF",
              "EEG C3-REF -- EEG P3-REF",
              "EEG P3-REF -- EEG O1-REF",
              "EEG FP2-REF -- EEG F4-REF",
              "EEG F4-REF -- EEG C4-REF",
              "EEG C4-REF -- EEG P4-REF",
              "EEG P4-REF -- EEG O2-REF",]

montage_22 = ["EEG FP1-REF -- EEG F7-REF",
              "EEG F7-REF -- EEG T3-REF",
              "EEG T3-REF -- EEG T5-REF",
              "EEG T5-REF -- EEG O1-REF",
              "EEG FP2-REF -- EEG F8-REF",
              "EEG F8-REF -- EEG T4-REF",
              "EEG T4-REF -- EEG T6-REF",
              "EEG T6-REF -- EEG O2-REF",
              "EEG A1-REF -- EEG T3-REF",
              "EEG T3-REF -- EEG C3-REF",
              "EEG C3-REF -- EEG CZ-REF",
              "EEG CZ-REF -- EEG C4-REF",
              "EEG C4-REF -- EEG T4-REF",
              "EEG T4-REF -- EEG A2-REF",
              "EEG FP1-REF -- EEG F3-REF",
              "EEG F3-REF -- EEG C3-REF",
              "EEG C3-REF -- EEG P3-REF",
              "EEG P3-REF -- EEG O1-REF",
              "EEG FP2-REF -- EEG F4-REF",
              "EEG F4-REF -- EEG C4-REF",
              "EEG C4-REF -- EEG P4-REF",
              "EEG P4-REF -- EEG O2-REF",]

ch_names_19 = ["EEG FP1-REF",
               "EEG FP2-REF",
               "EEG F3-REF",
               "EEG F4-REF",
               "EEG C3-REF",
               "EEG C4-REF",
               "EEG P3-REF",
               "EEG P4-REF",
               "EEG O1-REF",
               "EEG O2-REF",
               "EEG F7-REF",
               "EEG F8-REF",
               "EEG T3-REF",
               "EEG T4-REF",
               "EEG T5-REF",
               "EEG T6-REF",
               "EEG FZ-REF",
               "EEG CZ-REF",
               "EEG PZ-REF",]

ch_names_19_le = ["EEG FP1-LE",
                  "EEG FP2-LE",
                  "EEG F3-LE",
                  "EEG F4-LE",
                  "EEG C3-LE",
                  "EEG C4-LE",
                  "EEG P3-LE",
                  "EEG P4-LE",
                  "EEG O1-LE",
                  "EEG O2-LE",
                  "EEG F7-LE",
                  "EEG F8-LE",
                  "EEG T3-LE",
                  "EEG T4-LE",
                  "EEG T5-LE",
                  "EEG T6-LE",
                  "EEG FZ-LE",
                  "EEG CZ-LE",
                  "EEG PZ-LE",]

ch_names_21 = ["EEG FP1-REF",
               "EEG FP2-REF",
               "EEG F3-REF",
               "EEG F4-REF",
               "EEG C3-REF",
               "EEG C4-REF",
               "EEG P3-REF",
               "EEG P4-REF",
               "EEG O1-REF",
               "EEG O2-REF",
               "EEG F7-REF",
               "EEG F8-REF",
               "EEG T3-REF",
               "EEG T4-REF",
               "EEG T5-REF",
               "EEG T6-REF",
               "EEG A1-REF",
               "EEG A2-REF",
               "EEG FZ-REF",
               "EEG CZ-REF",
               "EEG PZ-REF",]

ch_names = ["FP1-F7",
            "F7-T3",
            "T3-T5",
            "T5-O1",
            "FP2-F8",
            "F8-T4",
            "T4-T6",
            "T6-O2",
            "T3-C3",
            "C3-CZ",
            "CZ-C4",
            "C4-T4",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",]


def load_eeg_file(file_path, montage=True, max_len=None):
    # Load
    try:
        f = pyedflib.EdfReader(file_path)
    except OSError:
        return None, None, None, None
    # Get original data
    signal_labels = f.getSignalLabels()
    reference = "ar"
    if "-LE" in " ".join(signal_labels):
        signal_labels = [signal_label.replace("-LE", "-REF") for signal_label in signal_labels]
        reference = "le"
    n_samples = f.getNSamples()
    orig_len = np.max(n_samples)
    srate = f.getSampleFrequency(n_samples.tolist().index(orig_len))
    # Filter signals by name
    ch_idxs = [idx for idx, signal_label in enumerate(signal_labels) if signal_label in ch_names_19]
    # Set max_len
    if (max_len is None) or (orig_len < max_len * srate):
        max_len = orig_len
    else:
        max_len = int(max_len * srate)
    sigbufs = np.zeros((len(ch_idxs), max_len))
    for idx, ch_idx in enumerate(ch_idxs):
        sigbufs[idx, :] = f.readSignal(ch_idx, n=max_len, digital=False)
    signal_labels = [signal_labels[ch_idx] for ch_idx in ch_idxs]
    if montage:
        sigbufs = set_montage(sigbufs, signal_labels)
        if sigbufs is None:
            raise ValueError(f"Could not find channel for file: {file_path}")
    return sigbufs, srate, ch_names, reference


def store_eeg_file(signals, srate, ch_names, save_path):
    signal_headers = pyedflib.highlevel.make_signal_headers(ch_names, sample_frequency=srate)
    header = pyedflib.highlevel.make_header(patientname="patient_x", gender="Female")
    pyedflib.highlevel.write_edf(f"{save_path}.edf", signals, signal_headers, header)


def set_montage(eeg, signal_labels):
    montaged = np.zeros((20, eeg.shape[1]))
    for idx, montage_label in enumerate(montage_20):
        (ch1, ch2) = montage_label.split(" -- ")
        try:
            ch1_idx = signal_labels.index(ch1)
        except ValueError:
            print(f"Could not find channel {ch1}")
            return None
        ch2_idx = signal_labels.index(ch2)
        montaged[idx, :] = eeg[ch1_idx, :] - eeg[ch2_idx, :]
    return montaged


def get_event_samples(annot):
    tolerance = 0
    df = pd.DataFrame(annot, columns=["ch_idx", "start", "end", "event"])
    df = df.sort_values(by=["start", "end", "ch_idx"], ascending=True, ignore_index=True)
    new_samples = []
    for event in [1, 2, 3, 4, 5, 6]:
        event_df = df.loc[df["event"] == event, :].copy()
        event_df.reset_index(inplace=True)
        if len(event_df) == 0:
            continue
        reduced_df = event_df.loc[[0], ["start", "end", "event"]].copy()
        for idx in range(1, len(event_df)):
            # Get the current interval
            current_interval = event_df.iloc[idx, :]
            # Get the last interval in the reduced DataFrame
            last_interval = reduced_df.iloc[-1, :]

            # Check if the current interval overlaps with the last interval
            if current_interval["start"] - tolerance <= last_interval["end"]:
                # Update the end_point of the last interval if needed
                if current_interval["end"] > last_interval["end"]:
                    reduced_df.at[last_interval.name, "end"] = current_interval["end"]
            else:
                # Append the current interval to the reduced DataFrame
                # reduced_df = reduced_df.append(current_interval[["start", "end", "event"]])
                reduced_df = pd.concat([reduced_df, current_interval[["start", "end", "event"]].to_frame().T])
        # Reset the index of the reduced DataFrame
        reduced_df = reduced_df.reset_index(drop=True)
        new_samples.append(reduced_df.values)
    try:
        new_samples = np.concatenate(new_samples, axis=0)
    except ValueError:
        return None
    return new_samples


def read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def parse_age_and_gender_from_edf_header(file_path):
    header = read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender


def load_annotation(file_path, use_montage_20=True):
    annot_path = file_path.split(".")[0] + ".rec"
    annot = pd.read_csv(annot_path, sep=",").values
    df = pd.DataFrame(annot, columns=["ch_idx", "start", "end", "event"])
    df = df.sort_values(by=["event", "start", "end", "ch_idx"], ascending=True, ignore_index=True)
    if use_montage_20:
        drop_idx = [8.0, 13.0]
        remap_idx = {9.0: 8.0,
                     10.0: 9.0,
                     11.0: 10.0,
                     12.0: 11.0,
                     14.0: 12.0,
                     15.0: 13.0,
                     16.0: 14.0,
                     17.0: 15.0,
                     18.0: 16.0,
                     19.0: 17.0,
                     20.0: 18.0,
                     21.0: 19.0}
        mask = ~df["ch_idx"].isin(drop_idx)
        df = df.loc[mask, :]
        df.replace({"ch_idx": remap_idx}, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

