import toml
import os
import numpy as np
import random
import torch
import shutil
from torch.utils.data import WeightedRandomSampler
from lightning.fabric import seed_everything
import time
import lightning.pytorch as pl
import logging

log = logging.getLogger("lightning.pytorch.core")


class TimeCallback(pl.Callback):
    """
    Custom callback to overwrite default method for training time tracking
    for individual epochs
    """

    def __init__(self, start):
        self.start = start

    def on_train_epoch_end(self, trainer, pl_module):
        train_time = np.round((time.time() - self.start) / 3600, 2)
        self.log("training_hrs", train_time, sync_dist=True, prog_bar=True, logger=True)


def time_track(start):
    """
    Get callback to track training time

    Returns
    -------
    callback: lightning.pytorch.callbacks.Callback
        A callback with custom functions to log training time
    """
    return TimeCallback(start)


def ds_to_sampler(
    dataset, resample_scale=1, resample_threshold=5, step=50, min=0, max=400, seed=None
):
    """
    Create weighted random sampler for data samples based on AGBD values.
    Samples with denser counts will be drawn less compare to those with sparse counts.
    The goal is to balance samples with different biomass density.

    Parameters
    ----------
    dataset : dataset object from the custom dataset class
        biomass dataset containing features and samples
    step: int
        step / bin widths for AGBD interval to count sample density (default 50)
    resample_scale: float
        scale for drawing samples relative to the dataset size (default 1, same as dataset size)
    min: int
        minimum biomass value for density estimation (default 0)
    max: int
        maximum biomass value for density estimation (default 400)

    Returns
    -------
    WeightedRandomSampler: Pytorch WeightedRandomSampler
        A sampler used for Pytorch data loader

    """
    # number of samples
    n = int(resample_scale * len(dataset))

    # get agb values for all samples in (sub-)dataset
    agb_list = [
        torch.amax((dataset[i][1]).flatten()).item() for i in range(len(dataset))
    ]
    # get bounds to estimate sample density
    bounds = np.arange(min, max + step, step)

    # no sampler if with single bin
    if len(bounds) == 1:
        return None

    # compute weights depends on sample density
    interval_weights = [
        1 / len(list(filter(lambda x: bounds[i] <= x <= bounds[i + 1], agb_list)))
        for i in range(len(bounds) - 1)
    ]

    # case: threshold set to limit image redundancy
    if resample_threshold:
        count = np.array([1 / i for i in interval_weights])  # sample count for bins
        resample_ratio = (n / len(count)) / np.array(count)  # scale of redundancy
        adjusted_scale = np.array(  # set special scale if redundancy exceeds threshold
            [
                1 if r <= resample_threshold else resample_threshold / r
                for r in resample_ratio
            ]
        )
        interval_weights *= adjusted_scale  # apply scale to samples

    # fit agb samples to its corresponding weights
    sample_weights = [
        interval_weights[int(np.min([agb // 50, len(interval_weights) - 1]))]
        for agb in agb_list
    ]

    assert len(sample_weights) == len(
        dataset
    ), "Unmatched lengths of dataset and weights!"

    # random sampling to balance agb values along spectrum
    if seed:
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=n, replacement=True, generator=seed
        )
    else:
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=n, replacement=True
        )


def load_config(config_path: str) -> dict:
    """Parses the provided configuration file and returns the content as a dictionary"""
    if not os.path.exists(config_path):
        raise ValueError("The provided config file doesn't exist")

    cfg = toml.load(config_path)

    return cfg


def copy_config(config_path, logdir):
    """Copies the config to the log dir for later reference"""
    shutil.copy2(config_path, os.path.join(logdir, "used_config.toml"))


def copy_src(src, logdir):
    """Copies source code current version for later reference"""
    shutil.copytree(src, os.path.join(logdir, "src"))


def create_toml(config_dict, logdir):
    dst_file = os.path.join(logdir, "used_config.toml")

    with open(dst_file, "w") as f:
        r = toml.dump(config_dict, f, encoder=toml.TomlNumpyEncoder())


def set_random_seeds(seed: int, use_deterministic_algorithms: bool = True):
    """Enable deterministic operations with random seeds for reproducibility"""
    if use_deterministic_algorithms:
        log.info(
            "Using CUDA deterministic algorithms (slower training but reproducible results"
        )
    else:
        log.info(
            "Using CUDA non-deterministic algorithms (faster training but non-reproducible results)"
        )
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    seed_everything(seed)


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
