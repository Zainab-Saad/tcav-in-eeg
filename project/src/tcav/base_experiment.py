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
    device = "cpu"

    # device = "cuda"

    print('=============device', device)

    print('=============the zip is', zip(concept_codes, concept_names))

    # Get input data
    input_data = get_dataset(input_code, random_n_samples=n_input)
    # temporarily change the batch size to 16
    n_input = 16
    input_data = torch.utils.data.DataLoader(input_data, batch_size=n_input, shuffle=False)
    input_data = next(iter(input_data)).to(device)

    print('=============done moving data to cuda')
    # Assemble experimental sets
    experimental_sets = []
    for n in range(1, n_runs + 1):  # Count from one
        print('================= n', n)
        experimental_set = []
        for concept_code, concept_name in zip(concept_codes, concept_names):
            concept_data = get_dataset(concept_code, random_n_samples=n_concept_sampels, random_state=n)
            concept_data = torch.utils.data.DataLoader(concept_data)
            # for batch in concept_data:
            #     batch = batch.to(device)  # Ensure concept data is on the same device
            concept = Concept(id=concept_code * 1000 + n, name=f"{concept_name}_{n:03d}", data_iter=concept_data)
            print('============concept', concept)
            experimental_set.append(concept)
        experimental_sets.append(experimental_set)

    print('==========start getting model prediction')
    # Get the model
    estimator = xceptiontime_v5(device=device)
    estimator.initialize()
    estimator.load_params(f_params=f"{checkpoint}params.pt")
    print('============loaded params')
    _ = estimator.predict(np.zeros([1, 20, 150], dtype="float32"))
    model = estimator.module_
    model.to(device)

    # print('======================model', model)
    print('done getting model prediction')


    print('start applying tcav')

    # Apply TCAV
    print(f"================================Dataset shape: {input_data.shape}")
    print(f"===========Input dataset device: {input_data.device}")
    print(f"===========Model parameters device: {next(model.parameters()).device}")

    tcav = TCAV(model=model, layers=layers,
                model_id=model_id, save_path=tcav_path, classifier=TCAVClassifier())

    print('========tcav', tcav)
    tcav_scores = tcav.interpret(inputs=input_data, experimental_sets=experimental_sets, target=target)
    tcav_scores = dict(tcav_scores)
    print('=========tcav_scores', tcav_scores)
    stats = {}
    for cav_key in tcav.cavs.keys():
        exp_stats = {}
        for layer_key in tcav.cavs[cav_key].keys():
            exp_stats[layer_key] = {"concepts": tcav.cavs[cav_key][layer_key].concepts,
                                    "layer": tcav.cavs[cav_key][layer_key].layer,
                                    "stats": {"classes": tcav.cavs[cav_key][layer_key].stats["classes"],
                                              "accs": tcav.cavs[cav_key][layer_key].stats["accs"]}}
            print('exp_stats: ', exp_stats)
        stats[cav_key] = exp_stats

    with open(f"{tcav_path}/{input_name}_{'_'.join(concept_names)}.pkl", "wb") as f:
        pickle.dump({"experimental_sets": experimental_sets,
                     "tcav_scores": tcav_scores,
                     "stats": stats}, f)
    print(f"Runtime: {time.time() - start_time:.2f} seconds.")

