import pathlib
from pathlib import Path
import os
import torch
import numpy as np
import random
import warnings
import torch
from sklearn.model_selection import train_test_split, KFold
import json

# config for wandb
def config(args):
    warnings.filterwarnings("ignore", category=UserWarning)

    #     OUTPUT_DIR: Path("./output")
    #     LOGS_DIR: Path(OUTPUT_DIR, "logs")
    #     MODEL_DIR: Path(OUTPUT_DIR, "models")
    #     LOGS_DIR.mkdir(parents=True, exist_ok=True)
    #     MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # }

# Seeding and reproducibility
def seed_all(seed: int = 42):
    """Seed all random number generators"""
    print("Using Seed Number {}".format(seed));

    os.environ["PYTHONHASHSEED"] = str(seed) # set PYTHONHASHSEED end var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed) #pytorch both CPU and CUDA
    np.random.seed(seed) # for numpy pseudo-random generators
    random.seed(seed) # for python built-in pseudo-random generators
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled  = True

def seed_worker(_worker_id):
    """Seed a worker with the given id"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_stacked_tensor(list_of_list_of_tensors):
    """Converts a list of list of tensors to a pytorch tensor"""
    stacked_tensor = torch.stack([torch.stack(sublist, dim=0) for sublist in list_of_list_of_tensors], dim=0)
    return stacked_tensor

def convert_list_to_tensor(list_of_tensors):
    """Convert a list of tensors to a 2D tensor"""
    list_of_tensors = torch.stack(list_of_tensors)
    return list_of_tensors

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def make_fold_files(args):
    folder = os.path.join(args.input_dir, args.text_input_dir)
    save_dir = os.path.join(folder, "s2_split{}".format(args.kfold))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(folder, args.text_file)
    file = open(file_path)
    data = json.load(file)

    data_trainval, data_test = train_test_split(data, test_size=0.1, random_state=args.seed)
    save_file = os.path.join(save_dir, 'test.json')
    with open(save_file, 'w') as f:
        json.dump(data_test, f)

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    fold = 1
    for train_idxs, val_idxs in kf.split(data_trainval):
        if torch.is_tensor(train_idxs):
            train_idxs = train_idxs.tolist()
        if torch.is_tensor(val_idxs):
            val_idxs = val_idxs.tolist()

        print(f"Saving fold {fold}")
        data_trainval = np.array(data_trainval)
        train_data = data_trainval[train_idxs]
        save_file = os.path.join(save_dir, 'fold{}_train.json'.format(fold))
        with open(save_file, 'w') as f1:
            json.dump(train_data.tolist(), f1)
        val_data = data_trainval[val_idxs]
        save_file = os.path.join(save_dir, 'fold{}_val.json'.format(fold))
        with open(save_file, 'w') as f2:
            json.dump(val_data.tolist(), f2)
        fold += 1
