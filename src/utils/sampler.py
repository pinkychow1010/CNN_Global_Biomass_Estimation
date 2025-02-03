# sampler.py

import numpy as np
from torch.utils.data import WeightedRandomSampler
import h5py
import copy


class Sampler:
    """
    Generate data sampler taken balanced AGB, land use and spatial locations into account.
    Create weighted random sampler for data samples based on aforementioned criteria.
    Samples with denser counts will be drawn less compare to those with sparse counts.
    The goal is to provide balance samples for CNN model training.

    Attributes
    ----------
    datafile : training hdf file, independent of the dataset class
        path with dataset containing AGB, land use and location features
    step: int
        step / bin widths for AGBD interval to count sample density (default 50)
    resample_scale: float
        scale for drawing samples relative to the dataset size (default 1, same as dataset size)
    min_agb: int
        minimum biomass value for density estimation (default 0)
    max_agb: int
        maximum biomass value for density estimation (default 400)
    grid_size:int
        number of divide along lat and lon to generate grid (default 5)
    redraw_thres: int
        maximum limit for sample redrawal. will be used to adjust sample weights. (default 5)
    seed: Generator
        seed to allow deterministic random sampling / bootstrapping (default None)
    location_bootstrap: bool
        balance spatial locations (lat & lon) when bootstrapping. (default True)
    agb_bootstrap: bool
        balance agb values when bootstrapping. (default True)
    landuse_bootstrap: bool
        balance land use at spatial grid when bootstrapping. (default True)
    xy_use: bool
        option to use location
    landuse_use: bool
        option to use landuse

    Methods
    -------
    compute_grid:
        assign grid identity for each sample based on centroid location
    setup:
        retrieve relevant data layers from hdf file
    compute_landuse_by_grid_weights:
        compute histogram for each grid and landuse to generate weights for each sample.
        aims to balance samples for each grid and each land use class within grid.
    compute_agb_weights:
        compute weights solely based on biomass values. aims to balance samples along agb spectrum.
    apply_threshold:
        adjust weights to limit too many redraw from the same sample, based on redraw_thres
    finalize_weights:
        combine computed weights for agb and land use. adjust with apply_threshold.
    __call__:
        full workflow for generating WeightedRandomSampler for model training.

    Returns
    -------
    WeightedRandomSampler: Pytorch WeightedRandomSampler
        A sampler used for Pytorch data loader

    """

    def __init__(
        self,
        datafile,
        data_idx,
        seed=None,
        step: int = 50,
        resample_scale: int = 1,
        min_agb: int = 0,
        max_agb: int = 500,
        grid_size: int = 5,
        redraw_thres: int = 5,
        location_bootstrap=True,
        agb_bootstrap=True,
        landuse_bootstrap=True,
    ):
        # error handling for no bootstrapping variables
        assert (
            location_bootstrap or agb_bootstrap or landuse_bootstrap
        ), "Variables not found for sampling!"

        self.file = datafile
        self.idx = data_idx
        self.step = step
        self.resample_scale = resample_scale
        self.min_agb = min_agb
        self.max_agb = max_agb
        self.redraw_thres = redraw_thres
        self.grid_size = grid_size
        self.seed = seed
        self.location_bootstrap = location_bootstrap
        self.agb_bootstrap = agb_bootstrap
        self.landuse_bootstrap = landuse_bootstrap

        # get variables from hdf file
        self.setup()

        # assign grid class for samples
        self.compute_grid()

        self.datasize = len(self.lat)  # original sample size
        self.sample_size = resample_scale * self.datasize  # resampled population

    def compute_grid(self):
        """get grid id for samples"""
        lat, lon = self.lat, self.lon
        # compute bins
        ybins = np.linspace(np.min(lat), np.max(lat), self.grid_size)
        xbins = np.linspace(np.min(lon), np.max(lon), self.grid_size)
        # assign samples to bins
        yclass = np.digitize(lat, ybins, right=True)
        xclass = np.digitize(lon, xbins, right=True)
        # turn bins into id
        self.loc_class = (1 + xclass) * 10 + yclass

    def revert_landuse_from_features(self, arr):
        """get land use class from one hot encoding"""
        mid_id = int(arr.shape[1] / 2)
        index = np.argmax(arr[-12:-1, mid_id, mid_id], axis=0).item()
        landuse_cls = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            95,
            100,
        ]

        return landuse_cls[index]

    def setup(self):
        """retrieve data layers from file"""

        with h5py.File(self.file, "r") as ds:
            self.landuse = ds["data"]["WorldCover"][:, 0, 16, 16][
                self.idx
            ]  # land cover class
            self.lat = ds["data"]["GEDI Lat Lon"][:, 0][self.idx]  # latitude
            self.lon = ds["data"]["GEDI Lat Lon"][:, 1][self.idx]  # longitude
            self.agb = np.squeeze(  # biomass
                np.nanmax(ds["data"]["AGBD Raster"][:, 0, :, :], axis=(1, 2))
            )[self.idx]

    def compute_spatial_grid_weights(self):
        """compute weights based on locations"""
        # estimate land use class distribution
        unique, counts = np.unique(self.loc_class, return_counts=True)
        sum = np.sum(counts)

        # compute weights from sample density in bins
        inverse_par = sum / counts
        weight_dict = {str(k): v for k, v in zip(list(unique), inverse_par)}

        # assign class weights to individual samples
        spatial_weights = [weight_dict[str(loc)] for loc in self.loc_class]
        self.spatial_weights = np.round(spatial_weights / np.min(spatial_weights), 5)

    def compute_landuse_by_grid_weights(self):
        """compute weights based on land use, defined by location"""

        loc_class = self.loc_class
        landcover = self.landuse

        # construct empty dict
        lulc_per_grid = dict.fromkeys(list(loc_class))

        # get land use distribution for each spatial grid
        for grid in np.unique(loc_class):
            landcover_grid = landcover[np.where(loc_class == grid)]
            unique, counts = np.unique(landcover_grid, return_counts=True)
            lulc_per_grid[grid] = dict(zip(unique, counts))

        # compute weights based on distribution above
        # store results to dict
        weight_dict = copy.deepcopy(lulc_per_grid)
        for d in lulc_per_grid:
            keys, values = weight_dict[d].keys(), weight_dict[d].values()
            arr = np.array(list(values))
            par = arr / np.sum(arr)  # percentage of total count

            weight = 1 / par  # weight as inverse density
            weight = np.round(
                weight * np.max(par), 5
            )  # normalize values with min 1 for weights
            weight_dict[d] = {k: v for k, v in zip(list(keys), weight)}  # update dict

        self.landuse_wdict = weight_dict

        # apply class weights to each sample
        weights = []
        for loc, luc in zip(loc_class, landcover):
            weight = self.landuse_wdict[loc][luc]
            weights.append(weight)

        # store final outputs
        self.landuse_by_grid_weights = weights

    def compute_agb_weights(self):
        """compute weights based on agb values"""

        # get bounds to estimate sample density
        bounds = np.arange(self.min_agb, self.max_agb + self.step, self.step)

        # no sampler if with single bin
        if len(bounds) == 1:
            return None

        # compute weights for each bin depends on sample density
        interval_weights = [
            1 / len(list(filter(lambda x: bounds[i] <= x <= bounds[i + 1], self.agb)))
            for i in range(len(bounds) - 1)
        ]

        # convert bin weights to sample weights
        sample_weights = np.array(
            [
                interval_weights[int(np.min([agb // 50, len(interval_weights) - 1]))]
                for agb in self.agb
            ]
        )

        # scale weights to avoid very small values
        sample_weights *= 1 / np.min(sample_weights)
        self.agb_weights = sample_weights

    def apply_threshold(self, sample_weights):
        """
        integrate redraw limits into weights.
        as minority class has limited sample, this avoids repeat sampling the same image for too many times
        """
        # max value for weights
        thres = np.sum(sample_weights) * (self.redraw_thres / self.datasize)
        final_weights = [w if w <= thres else thres for w in sample_weights]
        return final_weights

    def finalize_weights(self):
        """combine computed weights for agb and land use. adjust with apply_threshold."""
        # weight for biomass
        w1 = np.array(self.agb_weights)
        # weight for land use
        w2 = np.array(self.landuse_by_grid_weights)
        # weight for geo-cooredinates
        w3 = np.array(self.spatial_weights)

        return self.apply_threshold(w1 * w2 * w3)

    def __call__(self):
        """
        full workflow to create sampler:
        1) compute weights based on agb
        2) combine weights based on land use by location
        3) compute weights based on location
        4) combine all weights to single values
        5) apply threshold and get weights for individual samples
        6) generate sampler
        """

        # compute weights or not based on config
        # biomass bootstrapping
        if self.agb_bootstrap:
            self.compute_agb_weights()
        else:
            self.agb_weights = np.ones(self.datasize)

        # land cover bootstrapping
        if self.landuse_bootstrap:
            self.compute_landuse_by_grid_weights()
        else:
            self.landuse_by_grid_weights = np.ones(self.datasize)

        # location bootstrapping
        if self.location_bootstrap:
            self.compute_spatial_grid_weights()
        else:
            self.spatial_weights = np.ones(self.datasize)

        # combine all weights to single value for each sample
        self.final_weights = self.finalize_weights()

        assert (
            len(self.final_weights) == self.datasize
        ), "Unmatched lengths of dataset and weights!"

        # construct sampler from weights with replacement
        if self.seed:
            return WeightedRandomSampler(
                weights=self.final_weights,
                num_samples=self.sample_size,
                replacement=True,
                generator=self.seed,
            )
        else:
            return WeightedRandomSampler(
                weights=self.final_weights,
                num_samples=self.sample_size,
                replacement=True,
            )
