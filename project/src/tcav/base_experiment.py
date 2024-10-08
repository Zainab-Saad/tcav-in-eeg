import pickle
import torch
import numpy as np
import time
from captum.concept import Concept, TCAV
from src.ml.utils import xceptiontime_v5
from src.tcav.concept_classifier import TCAVClassifier
from src.tcav.utils import get_dataset


def run_tcav(input_code, input_name, concept_codes, concept_names, n_input=100, n_concept_sampels=50, n_runs=5,
             checkpoint=None, model_id="XceptionTimePlus", layers=None, tcav_path=None, target=1, device=None):
    start_time = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get input data
    input_data = get_dataset(input_code, random_n_samples=n_input)
    input_data = torch.utils.data.DataLoader(input_data, batch_size=n_input, shuffle=False)
    input_data = next(iter(input_data)).to(device)

    # Assemble experimental sets
    experimental_sets = []
    for n in range(1, n_runs + 1):  # Count from one
        experimental_set = []
        for concept_code, concept_name in zip(concept_codes, concept_names):
            concept_data = get_dataset(concept_code, random_n_samples=n_concept_sampels, random_state=n)
            concept_data = torch.utils.data.DataLoader(concept_data)
            concept = Concept(id=concept_code * 1000 + n, name=f"{concept_name}_{n:03d}", data_iter=concept_data)
            experimental_set.append(concept)
        experimental_sets.append(experimental_set)

    # Get the model
    estimator = xceptiontime_v5(device=device)
    estimator.initialize()
    estimator.load_params(f_params=f"{checkpoint}params.pt")
    _ = estimator.predict(np.zeros([1, 20, 150], dtype="float32"))
    model = estimator.module_
    model.to(device)

    # Apply TCAV
    tcav = TCAV(model=model, layers=layers,
                model_id=model_id, save_path=tcav_path, classifier=TCAVClassifier())
    tcav_scores = tcav.interpret(inputs=input_data, experimental_sets=experimental_sets, target=target)
    tcav_scores = dict(tcav_scores)
    stats = {}
    for cav_key in tcav.cavs.keys():
        exp_stats = {}
        for layer_key in tcav.cavs[cav_key].keys():
            exp_stats[layer_key] = {"concepts": tcav.cavs[cav_key][layer_key].concepts,
                                    "layer": tcav.cavs[cav_key][layer_key].layer,
                                    "stats": {"classes": tcav.cavs[cav_key][layer_key].stats["classes"],
                                              "accs": tcav.cavs[cav_key][layer_key].stats["accs"]}}
        stats[cav_key] = exp_stats

    with open(f"{tcav_path}/{input_name}_{'_'.join(concept_names)}.pkl", "wb") as f:
        pickle.dump({"experimental_sets": experimental_sets,
                     "tcav_scores": tcav_scores,
                     "stats": stats}, f)
    print(f"Runtime: {time.time() - start_time:.2f} seconds.")

