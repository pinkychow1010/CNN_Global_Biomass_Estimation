## Import Modules
# General Modules
import sys
import os
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
import numpy as np
import random
from matplotlib.colors import from_levels_and_colors
from sampler import Sampler

plt.rcParams.update({"font.size": 18})

# Deep Learning Modules
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
from torch import Generator

# Other scripts
current = os.path.dirname(os.path.realpath(__file__))  # run script from parent dir
sys.path.append(os.path.dirname(current))

from utils.datasets import BiomassDataset
from utils.processing_utils import load_config, set_random_seeds, copy_config
from models.baseline import BaseNeuralNetwork

# clear
os.system("clear")

# Extract last training session
LAST_SESSION = sorted(
    glob.glob("/home/pinkychow/biomass-estimation/training_logs/*"), reverse=True
)[0]

# Extract Checkpoint files created during last model training, Get the latest Checkpoint File
# PATH = "/home/pinkychow/biomass-estimation/training_logs/20230825_145939_baseline/epoch=19-val_loss=5.19.ckpt"
# PATH = "/home/pinkychow/biomass-estimation/training_logs/20230829_145234_test_elu/epoch=3-val_loss=5.13.ckpt"
PATH = "/home/pinkychow/biomass-estimation/training_logs/20230825_145939_baseline/epoch=9-val_loss=5.12.ckpt"
# max(glob.glob(os.path.join(LAST_SESSION, "*.ckpt")), key=os.path.getmtime)

print(
    "Checkpoint File used for prediction: \n",
    os.path.basename(PATH),
    ", from session ",
    LAST_SESSION,
)
print()

# Load Config File
base = os.path.basename(PATH)
cfg_path = PATH.replace(base, "used_config.toml")

print("Config File used: \n", cfg_path)
print()
cfg = load_config(cfg_path)

# Load saved checkpoint (latest checkpoint, latest epoch)
checkpoint = torch.load(PATH)
# print(checkpoint.keys())

# Initiate Base Model Object
model = BaseNeuralNetwork(
    in_channels=cfg["training"]["in_channels"],
    intermediate_depth=cfg["training"]["in_channels"] * 10,
    output_variance=cfg["training"]["output_variance"],
)

# remove model. from the dict keys to match expectations from load_state_dict()
# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/22
state_dict = torch.load(PATH)["state_dict"]
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("model.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

# construct an Optimizer with an iterable containing the model parameters and config options
optimizer = optim.Adam(
    model.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
)

# print(checkpoint['optimizer_states'][0])
optimizer.load_state_dict(checkpoint["optimizer_states"][0])

# Displayed saved info from model
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Switch to evaluation mode
model.eval()

# Call Entire Dataset with config options
dataset = BiomassDataset(cfg=cfg, mode="test")
# print(dataset.__dir__())
# print()

# Random retrieve subset for Testing (to be changed)
seed = Generator().manual_seed(cfg["training"]["seed"])

# Use same seed to retrieve test set (10%)
_, _, test_set = data.random_split(dataset, [0.7, 0.2, 0.1], generator=seed)

# Test Samples
print("Test Sample Count: ", len(test_set))

# Test in batches
batch_size = cfg["training"]["batch_size"]
print("Batch Size: ", batch_size)
print()


def data_split_with_idx(dataset, seed: int, ratio=[0.7, 0.2, 0.1]):
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


# (_, _, test_set), (_, _, test_idx) = data_split_with_idx(dataset, seed)

# test_sampler_object = Sampler(
#             cfg["dataset"]["path_train"],
#             test_idx,
#             resample_scale=1,  # no argmentation & no oversampling for validation
#             seed=seed,
#             redraw_thres=5,
#             location_bootstrap=True,  # training["location_bootstrap"],
#             agb_bootstrap=True,  # training["agb_bootstrap"],
#             landuse_bootstrap=True,  # training["landuse_bootstrap"],
#         )

# # call weighted random sampler
# test_sampler = test_sampler_object()

# # Create Data Loader for Testing
# test_loader = DataLoader(
#     test_set,
#     num_workers=0,
#     batch_size=batch_size,
#     shuffle=False,
#     sampler=test_sampler,
#     )

# Create Data Loader for Testing
test_loader = DataLoader(test_set, num_workers=0, batch_size=batch_size, shuffle=True)


# Function for computing RMSE
def compute_rmse(predicted_x, target):
    """
    Compute root mean square errors for prediction evaluation

    Parameters
    ----------
    predicted_x : tensor
        A series of predicted values from model based on inputs
    target: tensor
        Ground truth labels for all samples

    Returns
    -------
    loss: float
        Root Mean Square Error based on all predictions

    """
    n = torch.flatten(predicted).shape[0]
    loss = torch.sum(torch.square(predicted_x - target)) / n
    loss = torch.sqrt(loss)
    return loss


# Apply Model for Prediction

# Check performance for the top n batch
batch_n = 10

land = [
    "Tree cover",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare / sparse veg",
    "Snow and ice",
    "Water",
    "Wetland",
    "Mangroves",
    "Moss and lichen",
]

colors = [
    "green",  # Tree Cover
    "orange",  # Shrubland
    "yellow",  # Grassland
    "violet",  # Cropland
    "red",  # Built-up
    "gray",  # Bare / sparse vegetation
    "whitesmoke",  # Snow and ice
    "blue",  # Permanent water bodies
    "lightseagreen",  # Herbaceous wetland
    "springgreen",  # Mangroves
    "navajowhite",  # Moss and lichen
]

cmap, norm = from_levels_and_colors([0] + [val + 0.1 for val in range(0, 11)], colors)

avg_loss = []
with torch.no_grad():
    # Loop through batches
    for i, (inputs, outputs) in enumerate(test_loader):
        if i < batch_n:  # top n batch only
            # predict AGBD for each image
            predicted_outputs = model(inputs)
            predicted_outputs = predicted_outputs.to(torch.float32)

            # get mean predicted for each image (all pixels to single average value)
            mask = outputs < 0
            # predicted_unmask = predicted_outputs.detach().clone()
            # predicted_outputs[mask] = np.nan
            mask = torch.concatenate((mask, mask), axis=1)
            predicted_unmask = predicted_outputs.detach().clone()
            predicted_unmask = torch.unsqueeze(
                predicted_unmask[:, 0, :, :], 1
            )  # keep 4 dims
            predicted_outputs[mask] = np.nan

            # predicted agb mean
            predicted = torch.nanmean(
                torch.unsqueeze(predicted_outputs[:, 0, :, :], 1),  # keep 4 dims
                dim=(2, 3),
            )

            # get max AGB in each image
            outputs = outputs.to(torch.float32)
            outputs = torch.amax(outputs, dim=(2, 3))

            # compute loss using ground truth and mean prediction for proxy
            loss = compute_rmse(
                torch.flatten(torch.nan_to_num(predicted, nan=-9999)),
                torch.flatten(outputs),
            )

            # display mean predictions and ground truth labels for individual batch
            mean_agb = torch.mean(torch.flatten(outputs))
            mean_pred = torch.mean(torch.flatten(predicted))

            # display results
            print("Mean AGBD for the entire batch: ", mean_agb.item())
            print("Mean Prediction: ", mean_pred.item())

            # id for random samples within batch
            id = random.randint(0, batch_size - 3)

            # plotting
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            for i in range(id, id + 3):
                # compute ndvi for visualization
                b08 = inputs[i, 7, :, :]
                b04 = inputs[i, 3, :, :]
                # names = cfg["training"]["features"]["channel_names"]
                # b04 = cfg["training"]["features"]["channel_names"].index("Sentinel2_B04")
                # b08 = cfg["training"]["features"]["channel_names"].index("Sentinel2_B08")
                ndvi = (b08 - b04) / (b08 + b04)

                # land use
                # wc = [idx for idx, name in enumerate(names) if 'worldcover' in name]
                # landuse = torch.argmax(inputs[i, wc, :, :], dim=0)
                # major = land[torch.mode(landuse.flatten()).values.item()]
                if cfg["training"]["features"]["xy"]["use"]:
                    landuse = torch.argmax(inputs[i, -14:-3, :, :], dim=0)
                    major = land[torch.mode(landuse.flatten()).values.item()]
                else:
                    landuse = torch.argmax(inputs[i, -11:, :, :], dim=0)
                    major = land[torch.mode(landuse.flatten()).values.item()]
                # print(landuse.size())

                img0 = axes[0, i - id].imshow(
                    np.squeeze(predicted_unmask[i, :, :, :]),
                    # np.squeeze(predicted_outputs[i, :, :, :]),
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=200,
                )
                # display land use cover
                img1 = axes[1, i - id].imshow(
                    landuse, cmap=cmap, norm=norm, interpolation="none", aspect="auto"
                )
                # display NDVI
                img2 = axes[2, i - id].imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=1)

                # get labels
                target = int((torch.flatten(outputs))[i].item())
                prediction = int((torch.flatten(predicted))[i].item())
                avg_ndvi = np.round(torch.median(torch.flatten(ndvi)).item(), 2)
                # print prediction for plot
                axes[0, i - id].set_title(
                    f"Target AGB: {target}\nPrediction: {prediction}"
                )
                axes[1, i - id].set_title(f"Mainly:\n{major}")
                axes[2, i - id].set_title(f"Median NDVI: {avg_ndvi}")

                axes[0, i - id].axis("off")
                axes[1, i - id].axis("off")
                axes[2, i - id].axis("off")

            plt.show()

            print()
            print(f"Loss for batch {i}: ", loss.item())
            print()

            avg_loss.append(loss.item())
        else:
            break

print()
print(
    "Average Loss for all batches: ",
    np.round(np.sum(np.array(avg_loss)) / len(avg_loss), 2),
)
