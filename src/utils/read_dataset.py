# to run the model training:
# cd '/home/pinkychow/biomass-estimation/src/'
# python main.py '/home/pinkychow/biomass-estimation/src/configs/dev_config_basemodel_87k.toml'

# this script is used to inspect the performance of dataset class and develop relevant functionalities,
# eg. load checkpoint for prediction and resume model training

# import modules
import os
import sys
import numpy as np

# import from other scripts
current = os.path.dirname(os.path.realpath(__file__))  # run script from parent dir
sys.path.append(os.path.dirname(current))

from models.baseline import BaseNeuralNetwork
from utils.processing_utils import load_config, set_random_seeds, copy_config
from utils.training import BiomassModel
from utils.datasets import BiomassDataset
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

import pprint

os.system("clear")

# read config
config_path = "/home/pinkychow/biomass-estimation/training_logs/20230823_140330_default/used_config.toml"
# os.path.join(
#     os.path.expanduser("~"),
#     "biomass-estimation",
#     "src",
#     "configs",
#     "dev_config_basemodel_87k.toml",
# )
cfg = load_config(config_path)

# generate dataset
dataset = BiomassDataset(cfg=cfg)

pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(dataset.__dict__)

# first sample
feature, label = dataset[0]

print("AGB: ", torch.max(label.flatten()))
print("Shape of feature: ", dataset[1][0].shape)
print("Shape of label: ", dataset[1][1].shape)
print("Transform: ", dataset.transform)

# def ds_to_sampler(
#     dataset, resample_scale=1, resample_threshold=5, step=50, min=0, max=400
# ):
#     """
#     Create weighted random sampler for data samples based on AGBD values.
#     Samples with denser counts will be drawn less compare to those with sparse counts.
#     The goal is to balance samples with different biomass density.

#     Parameters
#     ----------
#     dataset : dataset object from the custom dataset class
#         biomass dataset containing features and samples
#     step: int
#         step / bin widths for AGBD interval to count sample density (default 50)
#     resample_scale: float
#         scale for drawing samples relative to the dataset size (default 1, same as dataset size)
#     min: int
#         minimum biomass value for density estimation (default 0)
#     max: int
#         maximum biomass value for density estimation (default 400)

#     Returns
#     -------
#     WeightedRandomSampler: Pytorch WeightedRandomSampler
#         A sampler used for Pytorch data loader

#     """
#     # number of samples
#     n = int(resample_scale * len(dataset))

#     # get agb values for all samples in (sub-)dataset
#     agb_list = [
#         torch.amax((dataset[i][1]).flatten()).item() for i in range(len(dataset))
#     ]
#     # get bounds to estimate sample density
#     bounds = np.arange(min, max + step, step)
#     # compute weights depends on sample density
#     interval_weights = [
#         1 / len(list(filter(lambda x: bounds[i] <= x <= bounds[i + 1], agb_list)))
#         for i in range(len(bounds) - 1)
#     ]

#     if resample_threshold:
#         count = np.array([1 / i for i in interval_weights])
#         resample_ratio = (n / len(count)) / np.array(count)
#         adjusted_scale = np.array(
#             [
#                 1 if r <= resample_threshold else resample_threshold / r
#                 for r in resample_ratio
#             ]
#         )
#         interval_weights *= adjusted_scale

#     # fit agb samples to its corresponding weights
#     sample_weights = [
#         interval_weights[int(np.min([agb // 50, len(interval_weights) - 1]))]
#         for agb in agb_list
#     ]

#     assert len(sample_weights) == len(
#         dataset
#     ), "Unmatched lengths of dataset and weights!"

#     # random sampling to balance agb values along spectrum
#     return WeightedRandomSampler(
#         weights=sample_weights, num_samples=n, replacement=True
#     )


# dataloader = DataLoader(dataset,
#                         batch_size=32,
#                         sampler=ds_to_sampler(dataset),
#                         shuffle=False
#                         )

# print(dataloader)
