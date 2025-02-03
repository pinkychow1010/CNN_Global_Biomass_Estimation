## Import Modules
# General Modules
import sys
import os
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
import random

plt.rcParams.update({"font.size": 18})

# Deep Learning Modules
import torch
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
from torch import Generator
from pytorch_lightning.loggers import CSVLogger

# Other scripts
current = os.path.dirname(os.path.realpath(__file__))  # run script from parent dir
sys.path.append(os.path.dirname(current))

from utils.datasets import BiomassDataset
from utils.processing_utils import load_config
from models.baseline import BaseNeuralNetwork
from utils.sampler import Sampler
from utils.training import BiomassModel
from utils.processing_utils import seed_worker


def main():
    """
    Full workflow to test CNN model
    """

    os.system("clear")
    # Extract last training session
    LAST_SESSION = sorted(
        glob.glob("/home/pinkychow/biomass-estimation/training_logs/*"), reverse=True
    )[0]

    # Extract Checkpoint files created during last model training, Get the latest Checkpoint File
    # (1) Option to use resampling for test data set
    resample = True

    # (2) Check Point File Path
    PATH = "/home/pinkychow/biomass-estimation/training_logs/20230829_145234_test_elu/epoch=19-val_loss=4.95.ckpt"
    # max(glob.glob(os.path.join(LAST_SESSION, "*.ckpt")), key=os.path.getmtime)

    # (3) Run functions
    test_model(checkpoint_path=PATH, resample=resample)


#########################
### Testing Functions ###
#########################


def data_split_with_idx(dataset, seed: int, ratio=[0.7, 0.2, 0.1]):
    """
    Split dataset & Get image samples id
    https://stackoverflow.com/questions/53532352/how-do-i-split-the-training-dataset-into-training-validation-and-test-datasets

    Parameters
    ----------
    dataset : Dataset class object
        custom dataset with from training data file with features and labels
    seed: int
        random seed for reproduction
    ratio: list
        ratio for splitting training, validation and testing set (in this order). Sum to 1.

    Returns
    -------
    (train_set, val_set, test_set)
        Dataset subset from torch.utils.data split for training, validation and testing (in this order).
    (train_idx, val_idx, test_idx)
        Image data index for for training, validation and testing (in this order).
        Can be used to slice the original hdf file.

    """
    # compute number of test/val elements
    n = len(dataset)
    n_train = int(n * ratio[0])
    n_valid = int(n * ratio[1])
    n_test = n - n_train - n_valid

    idx = list(range(n))  # indices to all elements

    if seed:
        random.seed(seed)

    random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting

    # indices for training, validation & testing set
    test_idx = idx[(n_train + n_test) :]

    # construct dataset subset
    test_set = Subset(dataset, test_idx)

    # return data & index
    return test_set, test_idx


def get_test_dataloader(cfg, dataset):
    """
    Split dataset & Get dataloader to load image samples

    Parameters
    ----------
    cfg : dict
        dict with config params used for training
    dataset: Dataset class object
        custom dataset from training data file

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
        loader for training samples with/without sampler
    val_loader: torch.utils.data.DataLoader
        loader for validation samples

    """
    # split the train set into training & validation
    training = cfg["training"]
    seed = Generator().manual_seed(training["seed"])

    test_set, test_idx = data_split_with_idx(dataset, training["seed"])

    test_sampler_object = Sampler(
        cfg["dataset"]["path_train"],
        test_idx,
        resample_scale=1,  # no argmentation & no oversampling for validation
        seed=seed,
        redraw_thres=training["redraw_thres"],
        location_bootstrap=True,
        agb_bootstrap=True,
        landuse_bootstrap=True,
    )
    # call weighted random sampler
    test_sampler = test_sampler_object()

    # testing dataset
    test_loader = DataLoader(
        test_set,
        num_workers=4,
        pin_memory=True,  # to save memory
        batch_size=training["batch_size"],
        sampler=test_sampler,
        shuffle=False,  # turn off shuffle for testing
        drop_last=True,
        worker_init_fn=seed_worker,  # deterministic
        generator=seed,
    )

    return test_loader


def test_model(
    checkpoint_path=PATH,
    model=BaseNeuralNetwork,
    dataset=BiomassDataset,
    resample=False,
):
    if resample:
        logdir = os.path.dirname(checkpoint_path)

        print(
            "Checkpoint File used for prediction: \n",
            os.path.basename(checkpoint_path),
            ", from session ",
            LAST_SESSION,
        )
        print()

        # Load saved checkpoint (latest checkpoint, latest epoch)
        checkpoint = torch.load(checkpoint_path)

        # Load Config File
        base = os.path.basename(checkpoint_path)
        cfg_path = checkpoint_path.replace(base, "used_config.toml")

        print("Config File used: \n", cfg_path)
        print()
        cfg = load_config(cfg_path)

        model = model(
            in_channels=cfg["training"]["in_channels"],
            intermediate_depth=cfg["training"]["in_channels"] * 10,
            output_variance=cfg["training"]["output_variance"],
        )

        # construct an Optimizer with an iterable containing the model parameters and config options
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        # Initiate Base Model Object
        dataset = dataset(cfg=cfg, mode="test")

        test_loader = get_test_dataloader(cfg, dataset)

        agbd_estimator = BiomassModel(model=model, cfg=cfg, ds=dataset)

        # remove model. from the dict keys to match expectations from load_state_dict()
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/22
        state_dict = torch.load(checkpoint_path)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        # print(checkpoint['optimizer_states'][0])
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])

        # Switch to evaluation mode
        model.eval()

        csv_logger = CSVLogger(save_dir=logdir, name="test_metrics")

        # construct customized trainer based on config
        trainer = pl.Trainer(
            max_epochs=cfg["training"]["epochs"],
            log_every_n_steps=cfg["training"]["lightning"]["log_every_n_batches"],
            default_root_dir=logdir,
            deterministic=cfg["training"]["cuda_deterministic_mode"],
            fast_dev_run=cfg["training"]["fast_dev_run"],
            check_val_every_n_epoch=1,
            logger=[csv_logger],  # log both to tensorboard and metrics.csv
        )

        trainer.test(
            model=agbd_estimator, dataloaders=test_loader, ckpt_path=checkpoint_path
        )

    else:
        logdir = os.path.dirname(checkpoint_path)

        print(
            "Checkpoint File used for prediction: \n",
            os.path.basename(checkpoint_path),
            ", from session ",
            LAST_SESSION,
        )
        print()

        # Load saved checkpoint (latest checkpoint, latest epoch)
        checkpoint = torch.load(checkpoint_path)

        # Load Config File
        base = os.path.basename(checkpoint_path)
        cfg_path = checkpoint_path.replace(base, "used_config.toml")

        print("Config File used: \n", cfg_path)
        print()
        cfg = load_config(cfg_path)

        model = model(
            in_channels=cfg["training"]["in_channels"],
            intermediate_depth=cfg["training"]["in_channels"] * 10,
            output_variance=cfg["training"]["output_variance"],
        )

        # construct an Optimizer with an iterable containing the model parameters and config options
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        # Initiate Base Model Object
        dataset = dataset(cfg=cfg, mode="test")

        agbd_estimator = BiomassModel(model=model, cfg=cfg, ds=dataset)

        # remove model. from the dict keys to match expectations from load_state_dict()
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/22
        state_dict = torch.load(checkpoint_path)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        # print(checkpoint['optimizer_states'][0])
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])

        # Switch to evaluation mode
        model.eval()

        # Call Entire Dataset with config options
        dataset = BiomassDataset(cfg=cfg, mode="test")

        # Random retrieve subset for Testing (to be changed)
        seed = Generator().manual_seed(cfg["training"]["seed"])

        # Use same seed to retrieve test set (10%)
        _, _, test_set = data.random_split(dataset, [0.7, 0.2, 0.1], generator=seed)

        # Test Samples
        print("Test Sample Count: ", len(test_set))

        # Create Data Loader for Testing
        test_loader = DataLoader(test_set, num_workers=0, batch_size=512, shuffle=True)

        csv_logger = CSVLogger(save_dir=logdir, name="test_metrics")

        # construct customized trainer based on config
        trainer = pl.Trainer(
            max_epochs=cfg["training"]["epochs"],
            log_every_n_steps=cfg["training"]["lightning"]["log_every_n_batches"],
            default_root_dir=logdir,
            deterministic=cfg["training"]["cuda_deterministic_mode"],
            fast_dev_run=cfg["training"]["fast_dev_run"],
            check_val_every_n_epoch=1,
            logger=[csv_logger],  # log both to tensorboard and metrics.csv
        )

        trainer.test(
            model=agbd_estimator, dataloaders=test_loader, ckpt_path=checkpoint_path
        )


if __name__ == "__main__":
    main()
