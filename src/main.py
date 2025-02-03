# to run the model training:
# cd '/home/pinkychow/biomass-estimation/src/'
# python main.py

# before git commit: black * to pass the job
# in case of out of memory error: reduce batch size by half in config (https://github.com/pytorch/pytorch/issues/16417)

# for tensorboard:
# tensorboard --logdir '/home/pinkychow/biomass-estimation/training_logs'

# kill existing tensorboard in port:
# lsof -i:6006 <- check pid
# kill -9 PID {pid}

# kill processes from own self
# pkill -U USERNAME

# tmux scroll up/down ctrl+b PgUp/PgDn

# TO DO: add output variance in config & parser (turn off for mc)
# implement mc model for main.py adapt for additional channel in output

# import modules
import os
import pprint
import time
import click
import sys
import random
import glob

from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
from torch import Generator
import lightning.pytorch as pl  # note: pytorch_lightning import with lightning.pytorch causes unexpected behaviors
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from datetime import datetime
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

import warnings

warnings.filterwarnings("ignore")

# import from other scripts
from models.baseline import BaseNeuralNetwork
from utils.processing_utils import (
    load_config,
    set_random_seeds,
    copy_config,
    copy_src,
    create_toml,
    time_track,
    seed_worker,
)
from utils.training import BiomassModel
from utils.datasets import BiomassDataset
from utils.config import Config
from utils.parser import ConfigParser
from utils.sampler import Sampler

# set up logger
import logging

log = logging.getLogger("lightning.pytorch.core")
pp = pprint.PrettyPrinter(width=24, compact=True)


# main controller for model train and rungit add
def main():
    """Main controller of model training"""
    start_time = time.time()  # start time for tracking reference

    print("\n\n▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
    log.info("⚡ Set up config & model.\n")
    print("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒\n\n")

    parser = ConfigParser()  # call parser to interpret user inputs
    cfg, logdir = setup(parser.parse_args())  # Load config to get dict

    # let user confirm displayed input config before continue
    if "resume_dir" not in cfg:
        if click.confirm("Do you want to continue?", default=False):
            pass
        else:
            sys.exit(0)
    else:
        previous_model = os.path.basename(os.path.normpath(cfg["resume_dir"]))
        if click.confirm(
            f"Do you want to resume training from {previous_model}?", default=False
        ):
            pass
        else:
            sys.exit(0)

    # ensure reproducibe random runs
    train_cfg = cfg["training"]
    set_random_seeds(train_cfg["seed"], train_cfg["cuda_deterministic_mode"])

    # Setup dataset
    dataset = BiomassDataset(cfg=cfg, mode=train_cfg["transform_mode"])

    # Get train and validation data loaders with random dataset split & sampler
    train_loader, val_loader, test_loader = get_dataloader(cfg, dataset)

    print("\n\n▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
    log.info("⚡ Building Model.")
    print("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒\n\n")

    # init the baseline model
    model = BaseNeuralNetwork(
        in_channels=train_cfg["in_channels"],
        intermediate_depth=train_cfg["in_channels"] * 10,
        output_variance=train_cfg["output_variance"],
    )
    agbd_estimator = BiomassModel(model=model, cfg=cfg, ds=dataset)

    # set up cnn trainer
    trainer = get_trainer(cfg, logdir, start_time)

    # in case of resume training from checkpoint file
    if "resume_dir" in cfg:
        log.info("Mode: resume training.")
        f_dir = cfg["resume_dir"]
        files = glob.glob(f"{f_dir}/*.ckpt")
        latest_ckpt_file = max(
            files, key=os.path.getctime
        )  # get the latest checkpoint to resume
        print(f"\nResume training from {latest_ckpt_file}.\n")

        trainer.fit(
            model=agbd_estimator,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=latest_ckpt_file,  # entry point
        )
    else:
        print("\n\n▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
        log.info("⚡ Starting training.")
        print("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒\n\n")

        # fit model for prediction from stratch
        trainer.fit(
            model=agbd_estimator,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    print("\n\n▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
    log.info("⚡ Starting testing.")
    print("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒\n\n")

    # testing after model fit
    trainer.test(model=agbd_estimator, dataloaders=test_loader, ckpt_path="best")


def setup(args_dict, print=True):
    """
    Set up the model with config copy and log directory

    Parameters
    ----------
    print : bool
        option to print config dict to console

    Returns
    -------
    cfg: dict
        A dict for config defined by config file / command line arguments
    logdir: str
        A path to training log file

    """
    # use config file input
    if "config" in args_dict:
        cfg_path = os.path.join("./configs", args_dict["config"])
        cfg = load_config(cfg_path)
    else:  # use config class
        cfg_obj = Config(**args_dict)  # default/ user inputs
        cfg = cfg_obj()  # __call__ to get dict

    # logging set up
    log_name = cfg["training"]["lightning"]["model_remark"]
    logdir = os.path.join(
        cfg["logging"]["logdir"],
        datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S_" + log_name),
    )
    os.makedirs(logdir, exist_ok=True)

    # store model train config params
    if "config" in args_dict:
        copy_config(cfg_path, logdir)  # copy file to new path
    else:
        create_toml(cfg, logdir)  # dump dict to toml

    # display config for model training
    if print:
        pp.pprint(cfg)

    # copy source code to log directory for later reference
    src_path = os.path.join(os.path.split(os.path.split(logdir)[0])[0], "src")
    copy_src(src_path, logdir)

    return cfg, logdir


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
    train_idx = idx[:n_train]
    val_idx = idx[n_train : (n_train + n_test)]
    test_idx = idx[(n_train + n_test) :]

    # construct dataset subset
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # return data & index
    return (train_set, val_set, test_set), (train_idx, val_idx, test_idx)


def get_dataloader(cfg, dataset):
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

    (train_set, val_set, test_set), (
        train_idx,
        val_idx,
        test_idx,
    ) = data_split_with_idx(dataset, training["seed"])

    # CASE: BOOTSTRAPPING
    if training["resample_scale"]:
        # shuffle performed by sampler instead
        shuffle = False

        print("\n\n▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
        log.info("⚡ Bootstrapping begins.")
        print("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒\n\n")
        ########################
        ### training sampler ###
        ########################
        # resample training subset using sampler
        train_sampler_object = Sampler(
            cfg["dataset"]["path_train"],
            train_idx,
            resample_scale=training["resample_scale"],
            seed=seed,
            redraw_thres=training["redraw_thres"],
            location_bootstrap=training["location_bootstrap"],
            agb_bootstrap=training["agb_bootstrap"],
            landuse_bootstrap=training["landuse_bootstrap"],
        )
        # call weighted random sampler
        train_sampler = train_sampler_object()

        ##########################
        ### validation sampler ###
        ##########################
        # resample validation subset
        val_sampler_object = Sampler(
            cfg["dataset"]["path_train"],
            val_idx,
            resample_scale=1,  # no argmentation & no oversampling for validation
            seed=seed,
            redraw_thres=training["redraw_thres"],
            location_bootstrap=True,
            agb_bootstrap=True,
            landuse_bootstrap=True,
        )
        # call weighted random sampler
        val_sampler = val_sampler_object()

    # CASE: NO BOOTSTRAPPING
    else:
        shuffle = True
        train_sampler, val_sampler = None, None

    ########################
    ### testing sampler  ###
    ########################
    # testing sampler is always the same for all experiment to ensure fair comparisons
    # resample testing subset
    print("\n\n▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
    log.info("⚡ Bootstrapping for test set begins.")
    print("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒\n\n")

    test_sampler_object = Sampler(
        cfg["dataset"]["path_train"],
        test_idx,
        resample_scale=1,  # no argmentation & no oversampling for validation
        seed=seed,
        redraw_thres=training["redraw_thres"],
        location_bootstrap=True,  # training["location_bootstrap"],
        agb_bootstrap=True,  # training["agb_bootstrap"],
        landuse_bootstrap=True,  # training["landuse_bootstrap"],
    )
    # call weighted random sampler
    test_sampler = test_sampler_object()

    # use params to construct data loader with/without sampler
    # training dataset
    train_loader = DataLoader(
        train_set,
        num_workers=4,
        pin_memory=True,  # to reduce memory use
        batch_size=training["batch_size"],
        sampler=train_sampler,  # bootstrapping with seeds to balance agb values in samples
        shuffle=shuffle,  # turn off shuffle with the use of random sampler
        drop_last=True,  # make sure sample size divided by batch size
        worker_init_fn=seed_worker,  # deterministic
        generator=seed,
    )
    # validation dataset
    val_loader = DataLoader(
        val_set,
        num_workers=4,
        pin_memory=True,  # to save memory
        batch_size=training["batch_size"],
        sampler=val_sampler,
        shuffle=False,  # turn off shuffle for validation
        drop_last=True,
        worker_init_fn=seed_worker,  # deterministic
        generator=seed,
    )
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

    return train_loader, val_loader, test_loader


def get_callback(cfg, logdir, start_time):
    """
    Callback for model depends on config

    Parameters
    ----------
    cfg: dict
        A dict for config defined by config file / command line arguments
    logdir: str
        A path to training log file

    Returns
    -------
    callback(s): lightning.pytorch.callbacks / list
        Either single lighning callbacks or a list of callbacks
    """
    # customize default callback to extract Validation Loss
    # https://stackoverflow.com/questions/58580442/get-totalloss-of-a-checkpoint-file
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=logdir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=cfg["training"]["lightning"][
            "save_top_k"
        ],  # get all epoch checkpoints
        every_n_epochs=cfg["training"]["lightning"]["checkpoint_interval"],
        verbose=True,
        monitor="val_loss",
        mode="min",  # min validation loss
    )

    # a customized callback to track time for each epoch
    training_time_callback = time_track(start_time)

    # a customized callback to monitor learning rate for learning rate schedulers during training
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # multiple callbacks: https://devblog.pytorchlightning.ai/introducing-multiple-modelcheckpoint-callbacks-e4bc13f9c185
    # early stopping: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping

    # multiple callbacks
    if cfg["training"]["lightning"]["early_stop"]:
        # define callback to stop training one validation loss bounce back
        earlystop_callback = EarlyStopping(
            monitor="val_loss", mode="min", patience=10, strict=True, verbose=True
        )
        return [
            checkpoint_callback,
            earlystop_callback,
            training_time_callback,
            lr_monitor,
        ]
    else:
        return [checkpoint_callback, training_time_callback, lr_monitor]


def get_trainer(cfg, logdir, start_time):
    """
    Callback for model depends on config

    Parameters
    ----------
    cfg: dict
        A dict for config defined by config file / command line arguments
    logdir: str
        A path to training log file

    Returns
    -------
    trainer: pl.Trainer
        Trainer defined by config params
    """

    # set up logger
    tb_logger = TensorBoardLogger(save_dir=logdir, name="lightning_logs", version=0)
    csv_logger = CSVLogger(save_dir=logdir, name="metrics")

    # construct customized trainer based on config
    return pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        log_every_n_steps=cfg["training"]["lightning"]["log_every_n_batches"],
        default_root_dir=logdir,
        callbacks=get_callback(
            cfg, logdir, start_time
        ),  # use custom checkpoint with valid loss, training time track and lr monitor
        deterministic=cfg["training"]["cuda_deterministic_mode"],
        fast_dev_run=cfg["training"]["fast_dev_run"],
        check_val_every_n_epoch=1,
        logger=[tb_logger, csv_logger],  # log both to tensorboard and metrics.csv
    )


if __name__ == "__main__":
    main()
