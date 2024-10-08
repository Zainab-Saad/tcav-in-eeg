import pandas as pd
from src.settings import tcav_name_dict_dir, cp_xceptiontime_v5, tcav_dir
from src.tcav.base_experiment import run_tcav

name_dict = pd.read_csv(tcav_name_dict_dir + "name_dict.csv", sep=",")
model_id = "XceptionTimePlus"
checkpoint = cp_xceptiontime_v5
layers = ["head.0", "head.1", "head.2", "head.3"]
experimental_sets = [
    ("spsw", "gped", "pled", "eyem", "artf", "bckg"),
    # ("random_eeg_delta", "random_eeg_theta", "random_eeg_alpha", "random_eeg_beta", "random_eeg_gamma"),
    # ("random_delta", "random_theta", "random_alpha", "random_beta", "random_gamma"),
    # ("male_eeg", "female_eeg"),
    # ("elderly_eeg", "young_eeg"),
    # ("spsw", "random_eeg"),
    # ("gped", "random_eeg"),
    # ("pled", "random_eeg"),
    # ("eyem", "random_eeg"),
    # ("artf", "random_eeg"),
    # ("bckg", "random_eeg"),
    # ("random_eeg_delta", "random_eeg"),
    # ("random_eeg_theta", "random_eeg"),
    # ("random_eeg_alpha", "random_eeg"),
    # ("random_eeg_beta", "random_eeg"),
    # ("random_eeg_gamma", "random_eeg"),
    # ("random_delta", "random_eeg"),
    # ("random_theta", "random_eeg"),
    # ("random_alpha", "random_eeg"),
    # ("random_beta", "random_eeg"),
    # ("random_gamma", "random_eeg"),
    # ("male_eeg", "random_eeg"),
    # ("female_eeg", "random_eeg"),
    # ("elderly_eeg", "random_eeg"),
    # ("young_eeg", "random_eeg"),
]
n_concept_samples = 75
input_name = "abnormal_test"
input_code = name_dict.loc[name_dict["Name"] == input_name, "Code"].values[0]
n_input = 100
n_runs = 100
tcav_path = tcav_dir
device = "cpu"
target = 1

for idx, exp in enumerate(experimental_sets):
    print(f"Experiment {idx+1} / {len(experimental_sets)}")
    concept_codes = []
    concept_names = exp
    for name in exp:
        concept_code = name_dict.loc[name_dict["Name"] == name, "Code"].values[0]
        concept_code = int(concept_code)
        concept_codes.append(concept_code)
    run_tcav(input_code, input_name, concept_codes, concept_names, n_input=n_input, n_concept_sampels=n_concept_samples,
             n_runs=n_runs, checkpoint=checkpoint, model_id=model_id, layers=layers, tcav_path=tcav_path, target=target,
             device=device)
