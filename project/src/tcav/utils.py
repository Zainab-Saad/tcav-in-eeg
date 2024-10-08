from src.dataset.cropped_dataset import CroppedDataset
from src.dataset.event_dataset import EventDataset
from src.dataset.filter_dataset import FilterDataset
from src.dataset.generated_dataset import GeneratedDataset
from src.settings import tuab_150hz_dir, tueg_150hz_dir, tuev_150hz_dir

tuab_root = tuab_150hz_dir
tueg_root = tueg_150hz_dir
tuev_root = tuev_150hz_dir


def get_dataset(concept_code, random_n_samples=10, random_state=0):
    dataset = None
    digits = [int(d) for d in str(concept_code)]

    # Placeholder
    dataset_root = None
    redefine_classes = None
    file_list = None
    freqs = None

    # Check TUAB index
    if digits[0] > 1:
        dataset_root = tuab_root
        if digits[0] == 2:
            # Normal
            redefine_classes = {0: 0}
            file_list = "train_list"
        if digits[0] == 3:
            # Normal
            redefine_classes = {0: 0}
            file_list = "eval_list_first_min"
        if digits[0] == 4:
            # Abnormal
            redefine_classes = {1: 1}
            file_list = "train_list"
        if digits[0] == 5:
            # Abnormal
            redefine_classes = {1: 1}
            file_list = "eval_list_first_min"
        dataset = CroppedDataset(root=dataset_root, file_list=file_list,
                                 random_n_samples=random_n_samples, redefine_classes=redefine_classes, return_target=False,
                                 random_state=random_state,
                                 buffer=False)
        return dataset

    # Check if TUEG index
    if digits[1] >= 1:
        dataset_root = tueg_root
        redefine_classes = None
        if digits[1] == 1:
            file_list = "train_list_random"
            # Frequency filtering?
            if digits[3] == 2:
                freqs = [1, 3]
            if digits[3] == 3:
                freqs = [4, 7]
            if digits[3] == 4:
                freqs = [8, 13]
            if digits[3] == 5:
                freqs = [31, 50]
        if digits[1] == 2:
            file_list = "train_list_male"
        if digits[1] == 3:
            file_list = "train_list_female"
        if digits[1] == 4:
            file_list = "train_list_old"
        if digits[1] == 5:
            file_list = "train_list_young"
        if freqs is None:
            dataset = CroppedDataset(root=dataset_root, file_list=file_list,
                                     random_n_samples=random_n_samples, redefine_classes=redefine_classes,
                                     return_target=False,
                                     random_state=random_state,
                                     buffer=False)
            return dataset
        else:
            dataset = FilterDataset(root=dataset_root, file_list=file_list,
                                    random_n_samples=random_n_samples, redefine_classes=redefine_classes,
                                    return_target=False,
                                    random_state=random_state,
                                    buffer=False,
                                    srate=150, freqs=freqs)
            return dataset

    # Check TUEV index
    if digits[2] >= 1:
        dataset_root = tuev_root
        file_list = "train_list"
        redefine_classes = {digits[2]: digits[2]}
        dataset = EventDataset(root=dataset_root, file_list=file_list,
                               random_n_samples=random_n_samples, redefine_classes=redefine_classes,
                               return_target=False,
                               random_state=random_state,
                               buffer=False)
        return dataset

    # Check Generated index
    if digits[3] >= 1:
        # Frequency filtering?
        if digits[3] == 2:
            freqs = [[1, 3]]
        if digits[3] == 3:
            freqs = [[4, 7]]
        if digits[3] == 4:
            freqs = [[8, 13]]
        if digits[3] == 5:
            freqs = [[31, 50]]
        dataset = GeneratedDataset(dataset_size=10, timeseries_len=9000, n_channels=20, random_seed=0,
                                   amplitude=[5, 25], srate=150, freqs=freqs, randomize_per_channel=False,
                                   return_target=False)
        return dataset

    return dataset
