import h5py
import numpy as np
import matplotlib.pyplot as plt
import copy
import pprint

pp = pprint.PrettyPrinter(width=30, compact=True)

PATH = "/home/pinkychow/biomass-estimation/data/biomass_dataset_train_87k.h5"


def descend_obj(obj, sep="\t"):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
        for key in obj.keys():
            print(sep, "-", key, ":", obj[key])
            descend_obj(obj[key], sep=sep + "\t")
    elif type(obj) == h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print(sep + "\t", "-", key, ":", obj.attrs[key])


def h5dump(path, group="/"):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path, "r") as f:
        descend_obj(f[group])


def compute_weights(loc_class, landuse, weight_dict):
    """
    generate weights for samples
    """
    weights = []
    for loc, luc in zip(loc_class, landuse):
        weight = weight_dict[loc][luc]
        weights.append(weight)
    return weights


h5dump(PATH)


with h5py.File(PATH, "r") as ds:
    lat = ds["data"]["GEDI Lat Lon"][:, 0]
    lon = ds["data"]["GEDI Lat Lon"][:, 1]
    print("Lat Range: ", np.min(lat), np.max(lat))
    print("Lon Range: ", np.min(lon), np.max(lon))

    ybins = np.linspace(np.min(lat), np.max(lat), 5)
    xbins = np.linspace(np.min(lon), np.max(lon), 5)
    # print(xbins)
    # print(ybins)

    yclass = np.digitize(lat, ybins, right=True)
    xclass = np.digitize(lon, xbins, right=True)
    # print(xclass)
    # print(yclass)

    loc_class = (1 + xclass) * 10 + yclass
    # print(loc_class)

    unique, counts = np.unique(loc_class, return_counts=True)
    loc_dict = dict(zip(unique, counts))
    print("Spatial Distribution for Each Grid: ")
    pp.pprint(loc_dict)

    landcover = ds["data"]["WorldCover"][:, 0, 16, 16]
    # print(landcover)
    unique, counts = np.unique(landcover, return_counts=True)
    lulc_dict = dict(zip(unique, counts))
    print("Land Use Distribution: ")
    pp.pprint(lulc_dict)

    lulc_per_grid = dict.fromkeys(list(loc_class))
    for grid in np.unique(loc_class):
        landcover_grid = landcover[np.where(loc_class == grid)]
        unique, counts = np.unique(landcover_grid, return_counts=True)
        lulc_per_grid[grid] = dict(zip(unique, counts))

    pp.pprint(lulc_per_grid)

    weight_dict = copy.deepcopy(lulc_per_grid)
    for d in lulc_per_grid:
        keys, values = weight_dict[d].keys(), weight_dict[d].values()
        arr = np.array(list(values))
        par = arr / np.sum(arr)
        weight = 1 / par
        weight = np.round(weight * np.max(par), 5)
        weight_dict[d] = {k: v for k, v in zip(list(keys), weight)}

    # pp.pprint(weight_dict)
    sample_weights = compute_weights(loc_class, landcover, weight_dict)
    print(sample_weights)

    # apply threshold: thres = np.sum(sample_weights)*(max_repeat/sample_size)
    max_repeat = 5
    sample_size = len(sample_weights)
    thres = np.sum(sample_weights) * (max_repeat / sample_size)
    final_weights = [w if w <= thres else thres for w in sample_weights]
