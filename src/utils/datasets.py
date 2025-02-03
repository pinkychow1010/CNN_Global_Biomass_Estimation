# import modules
from torch.utils.data import Dataset
from utils.transformer import Transformer
import torch
import h5py
import numpy as np
import logging
import math

np.seterr(divide="ignore", invalid="ignore")
logging.getLogger("__name__")

# To do:
# 1) add geolocations (in progress)
# 2) add different transform for train and valid (completed)
# 3) add spectral indices for training (completed)
# 4) add spectral indices in config (completed)
# 5) add transform in config (completed)
# 6) add weighted random sampler (in progress)
# 7) improve error handling in scripts
# 8) early stopping & training time tracking (completed)
# 9) learning rate scheduler
# 10) bootstrapping (in progress)
# 11) monte carlo dropout


class BiomassDataset(Dataset):  # inherent pytorch Dataset
    """
    Biomass dataset class to represent all training samples and labels used for CNN model training.

    Attributes
    ----------
    cfg: dict
        config for model training loaded
    file: str
        file path to training h5 file
    debug: bool
        debug mode option
    mode: str
        option for data augmentation ("test","crop_only","default")
    single_debug_idx: int
        debugsample index for debug mode and None otherwise
    data_size: int
        count of data samples in training file
    img_size: tuple
        2d shape of the individual image sample
    transform: torchvision.transforms.Compose(transforms)
        transformation steps for samples
    random_crop: bool
        random cropping option
    dataset: dict
        dict with all dataset stored with its attributes
    copdem: dict
        DEM features and config
    sentinel2: dict
        DEM features and config
    worldcover: dict
        Land use features and config
    agbd: dict
        Biomass features and config

    Methods
    -------
    auto_transform:
        Generate data sample transformation steps from torchvision, called in __getitem__().
    generate_feature_dict
        Generate dict for all dataset features and set to object attributes.
    dataset_dict
        Create dict for individual data variables.
    prepare_features
        Set up each data variables based on config and update attributes accordingly.
    setup_s2
        Set up Sentinel-2 features for all spectral bands and set to object attributes.
    setup_copdem
        Set up CopDEM: slope + altitude features and update attributes.
    setup_wc
        Set up World Cover Land Use features and update attributes.
    get_stats
        Retrieve mean and standard deviation from all features and update attributes accordingly.
    get_img_shape
        Extract image shape from individual samples of the dataset
    __len__
        Required function for custom Pytorch dataset class, used to count all samples in dataset.
    __getitem__
        Required function for custom Pytorch dataset class, used to retrieve single sample from dataset.
        Define iterable object to loop through training data samples with all features.
    standardise_features
        Retrieve single sample for single dataset with standardization
        bases on given means and stds keys in attributes.
    revert_feature_standardization
        Reverts standardisation for plotting batched image data in tensorboard log.
    get_one_hot_encoding
        Returns an one-hot encoded array with one channel for each class.
    fold_worldcover
        Creates a single channel array containing the ESA WorldCover classes from a one-hot encoded array.
    """

    def __init__(self, cfg: dict, mode: bool = "default"):
        """
        Constructs attributes from config for preprocessing and data transformation.

        Parameters
        ----------
            config: dict
                config loaded as dict
        """
        self.cfg = cfg  # store config dict
        self.mode = mode
        self.file = self.cfg["dataset"]["path_train"]  # file path to training data
        self.debug = cfg["training"]["single_sample_debug"]  # option for debug mode
        self.single_debug_idx = (  # sample id for debug mode
            self.cfg["training"]["single_sample_debug_idx"] if self.debug else None
        )
        self.data_size = len(self)  # training sample count
        self.img_shape = self.get_img_shape()  # original image sample size
        self.transform = self.init_transform()  # option for data augmentation
        self.random_crop = (  # random crop augmentation option
            cfg["training"]["use_random_crop"] and not self.debug
        )

        self.generate_feature_dict()  # dict to store feature for all data variables
        self.dataset = {  # dict to store locations of dataset info
            "CopDEM": self.copdem,
            "Sentinel-2 L2A": self.sentinel2,
            "WorldCover": self.worldcover,
            "AGBD Raster": self.agbd,
        }

        self.prepare_features()  # set up features for model

    def get_img_shape(self):
        """extract image shape"""
        with h5py.File(self.file, "r") as ds:
            shape = ds["data"]["Sentinel-2 L2A"][0, 0, :, :].shape
            assert (
                shape[0] == shape[1]
            ), "Pixel size of samples in x and y direction should be the same!"
            return shape

    def init_transform(self):
        """call transformer to set up data augmentation options"""
        return Transformer(self.cfg, self.mode)()

    def generate_feature_dict(self):
        """
        Generate dict for all dataset features and set to object attributes

        Returns
        -------
        None
        """
        # define empty dict for all datasets
        self.copdem = self.dataset_dict()
        self.sentinel2 = self.dataset_dict()
        self.worldcover = self.dataset_dict()
        self.agbd = self.dataset_dict()
        self.agbd["use"] = True

        # set up dict for lat lon in model training
        self.xy = self.dataset_dict()
        self.xy["use"] = self.cfg["training"]["features"]["xy"]["use"]
        self.xy["count"] = 4 if self.xy["use"] else 0

        # class for world cover land use type
        self.worldcover["class"] = [
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

    def dataset_dict(self):
        """
        Create dict for individual data variables

        use: whether to use dataset for model training
        count: count of dataset features to use
        feature: collection of dataset features
        index: order for features
        mean: mean value for each feature across all training samples
        std: standard deviation for each feature across all training samples

        Returns
        -------
        dataset_dict: dict
            a dict with predefined keys and empty values
        """
        return dict.fromkeys(["use", "count", "feature", "index", "mean", "std"])

    def prepare_features(self):
        """
        Set up each data variables based on config and update attributes accordingly

        Returns
        -------
        None
        """
        self.setup_s2()  # sentinel-2
        self.setup_copdem()  # DEM
        self.setup_wc()  # ESA world cover
        self.get_stats()  # for all features

    def setup_s2(self):
        """
        Set up Sentinel-2 features for all spectral bands and set to object attributes

        Returns
        -------
        None
        """

        # set up sentinel-2 bands to be used base on config
        sentinel2_features = self.cfg["training"]["features"]["sentinel2"]
        sentinel2_features_to_use = [
            b for b in sentinel2_features if sentinel2_features[b]["use"]
        ]

        self.sentinel2["use"] = True
        self.sentinel2["count"] = len(sentinel2_features_to_use)

        # sentinel 2 feature id
        s2_indices = [sentinel2_features[f]["index"] for f in sentinel2_features_to_use]
        # mean for individual scene
        s2_means = [sentinel2_features[f]["mean"] for f in sentinel2_features_to_use]
        # std for individual scene
        s2_stds = [sentinel2_features[f]["std"] for f in sentinel2_features_to_use]

        # Make sure the indices are sorted in increasing order (band 1 ..> band 12)
        self.sentinel2["features"] = sorted(
            zip(s2_indices, s2_means, s2_stds, sentinel2_features_to_use)
        )

        (
            self.sentinel2["index"],
            self.sentinel2["mean"],
            self.sentinel2["std"],
            self.sentinel2["feature_name"],
        ) = zip(*self.sentinel2["features"])

    def setup_copdem(self):
        """
        Set up CopDEM: slope + altitude features and update attributes

        Returns
        -------
        None
        """
        # fill feature dict for dem
        copdem_features = self.cfg["training"]["features"]["copdem"]
        copdem_features_to_use = [
            b for b in copdem_features if copdem_features[b]["use"]
        ]

        self.copdem["count"] = len(copdem_features_to_use)
        self.copdem["mean"] = [
            copdem_features[f]["mean"] for f in copdem_features_to_use
        ]
        self.copdem["std"] = [copdem_features[f]["std"] for f in copdem_features_to_use]

        self.copdem["index"] = [
            copdem_features[f]["index"] for f in copdem_features_to_use
        ]
        self.copdem["use"] = bool(self.copdem["index"])

    def setup_wc(self):
        """
        Set up World Cover Land Use features and update attributes

        Returns
        -------
        None
        """
        # fill feature dict for land use class layer
        self.worldcover["index"] = []
        self.worldcover["feature"] = self.cfg["training"]["features"]["worldcover"]

        self.worldcover["use"] = bool(self.worldcover["feature"]["use"])
        if self.worldcover["use"]:
            self.worldcover["index_start"] = (
                self.xy["count"] + self.sentinel2["count"] + self.copdem["count"]
            )

    def get_stats(self):
        """
        Retrieve mean and standard deviation from all features and update attributes accordingly

        Returns
        -------
        None
        """

        self.means = list(self.sentinel2["mean"])  # mean and std attribure
        self.stds = list(self.sentinel2["std"])

        if self.copdem["use"]:
            self.means += self.copdem["mean"]  # additional bands for all samples
            self.stds += self.copdem["std"]

    def compute_ndvi(self, data):
        """
        Calculate ndvi from Sentinel-2 data array using raw spectral values (non-standardised)
        """
        # compute index: Normalized difference vegetation index
        b08_id = self.sentinel2["feature_name"].index("b08")
        b04_id = self.sentinel2["feature_name"].index("b04")
        nir = data[b08_id, :, :]
        red = data[b04_id, :, :]

        # apply formula
        ndvi = (nir - red) / (nir + red)

        # missing value removal
        # not removing nan will cause invalid prediction in model (https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/11)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        return np.expand_dims(ndvi, axis=0)

    def compute_ndwi(self, data):
        """
        Calculate ndwi from Sentinel-2 data array using raw spectral values (non-standardised)
        """
        # compute index: Normalized difference water index
        # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
        b08_id = self.sentinel2["feature_name"].index("b08")
        b03_id = self.sentinel2["feature_name"].index("b03")
        b08 = data[b08_id, :, :]
        b03 = data[b03_id, :, :]

        # apply formula
        ndwi = (b03 - b08) / (b03 + b08)

        # missing value removal
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)
        return np.expand_dims(ndwi, axis=0)

    def compute_lai(self, data):
        """
        Calculate lai from Sentinel-2 data array using raw spectral values (non-standardised)
        """
        # compute index: Leaf Area index
        b08_id = self.sentinel2["feature_name"].index("b08")
        b04_id = self.sentinel2["feature_name"].index("b04")
        b02_id = self.sentinel2["feature_name"].index("b02")

        nir = data[b08_id, :, :]
        red = data[b04_id, :, :]
        blue = data[b02_id, :, :]

        # apply formula
        # https://www.indexdatabase.de/search/?s=EVI
        evi = 2.5 * (nir - red) / ((nir + 6 * red - 7.5 * blue) + 1)
        lai = 3.618 * evi - 0.118
        lai = np.nan_to_num(
            lai, nan=0.0, posinf=0.0, neginf=0.0
        )  # important: removing nan
        return np.expand_dims(lai, axis=0)

    def __len__(self):
        """
        Required function for custom Pytorch dataset class, used to count all samples in dataset

        Returns
        -------
        count: int
            Number of data samples in dataset
        """
        with h5py.File(self.file, "r") as ds:
            return ds["data"]["Sentinel-2 L2A"].shape[0]

    def __getitem__(self, idx):
        """
        Required function for custom Pytorch dataset class, used to retrieve single sample from dataset.
        Define iterable object to loop through training data samples with all features

        Parameters
        ----------
        idx: int
            index of data sample

        Returns
        -------
        out_features, out_agbd: Tuple
            Single sample (3d features and 3d biomass ground truth label as tensors)
        """
        # debug model option
        if self.debug:
            idx = self.single_debug_idx

        # print("Get ID: ", idx)
        with h5py.File(self.file, "r") as ds:
            # preprocessing for individual data variables
            ################
            ## Sentinel-2 ##
            ################
            # feature standardization using mean and std
            # exclude vegetation indices
            self.sentinel2["item"] = self.standardise_features(
                ds["data"], "Sentinel-2 L2A", idx
            )

            # compute vegetation indices on-the-fly
            # index 12: Normalized difference vegetation index (NDVI)
            if "ndvi" in self.sentinel2["feature_name"]:
                ndvi = self.compute_ndvi(ds["data"]["Sentinel-2 L2A"][idx, :, :, :])
                self.sentinel2["item"] = np.concatenate([self.sentinel2["item"], ndvi])

            # index 13: Leaf area index (LAI)
            if "lai" in self.sentinel2["feature_name"]:
                lai = self.compute_lai(ds["data"]["Sentinel-2 L2A"][idx, :, :, :])
                self.sentinel2["item"] = np.concatenate([self.sentinel2["item"], lai])

            # index 14: Normalized Difference Water Index (NDWI)
            if "ndwi" in self.sentinel2["feature_name"]:
                ndwi = self.compute_ndwi(ds["data"]["Sentinel-2 L2A"][idx, :, :, :])
                self.sentinel2["item"] = np.concatenate([self.sentinel2["item"], ndwi])

            assert (
                self.sentinel2["item"].shape[0] == self.sentinel2["count"]
            ), "Sentinel-2 features do not match feature count!"

            ################
            ##   COPDEM   ##
            ################
            # normalize range
            if self.copdem["use"]:
                self.copdem["item"] = self.standardise_features(
                    ds["data"], "CopDEM", idx
                )

            #################
            ## World Cover ##
            #################
            # convert categorical feature (World Cover) to one-hot encoding
            if self.worldcover["use"]:
                self.worldcover["item"] = self.get_one_hot_encoding(
                    ds["data"]["WorldCover"][idx, 0, :, :]
                )

            #################
            ##   Location  ##
            #################
            if self.xy["use"]:
                # prepare spatial location channels
                lat = ds["data"]["GEDI Lat Lon"][idx, 0]
                lon = ds["data"]["GEDI Lat Lon"][idx, 1]

                # transform latitude
                lat1 = np.sin(lat * (math.pi / 180))
                lat2 = np.cos(lat * (math.pi / 180))

                # transform longitude
                lon1 = np.sin(lon * (math.pi / 180))
                lon2 = np.cos(lon * (math.pi / 180))

                # convert 1d value to 2d data
                getImg = lambda a: np.full(self.img_shape, a)[np.newaxis, :]

                # stack channels
                self.xy["item"] = np.concatenate(
                    [getImg(lat1), getImg(lat2), getImg(lon1), getImg(lon2)]
                )

            # collect all features based on config
            features = np.concatenate(
                [var["item"] for var in self.dataset.values() if "item" in var]
            )

            if self.xy["use"]:
                features = np.concatenate(
                    [self.xy["item"], features]
                )  # location as the first layers
            # print(features.shape)

            ###############
            ## Biomass   ##
            ###############
            # collect biomass labels
            agbd = ds["data"]["AGBD Raster"][idx, :, :, :]

        # data augmentation
        if self.transform:
            # concatanate predictors and labels to ensure same random params in augumentation
            sample = self.transform(torch.Tensor(np.concatenate([agbd, features])))

            # separate predictors and labels for outputs
            out_agbd = torch.Tensor(
                sample[0, :, :]
            )  # unsqueeze biomass to ensure same shape as predictors

            out_features = torch.Tensor(sample[1:, :, :])

            return out_features, torch.unsqueeze(
                out_agbd, 0
            )  # single sample feature and label

        else:  # without data augmentation
            return torch.Tensor(features), torch.Tensor(agbd)

    def standardise_features(self, data: np.array, name: str, idx: int):
        """
        Retrieve single sample for single dataset with standardization
        bases on given means and stds keys in attributes

        Parameters
        ----------
        data: dict
            ds["data"] with keys of different datasets
        name: str
            name of the dataset, stored inside object dataset attribute
            One of: "CopDEM", "Sentinel-2 L2A", "WorldCover", "AGBD Raster"
        idx: int
            index of sample to be retrieved for standardization

        Returns
        -------
        standardized_features: np.array
            Single sample of single variable standardized
        """
        # get data variable
        data_dict = self.dataset[name]

        if name == "Sentinel-2 L2A":  # special case for sentinel 2
            # get raw bands only (start with b); exclude vegetation indices
            count, index = [], []
            for i in list(self.dataset[name]["feature_name"]):
                if i.startswith("b"):
                    id = self.dataset[name]["feature_name"].index(i)
                    index.append(data_dict["index"][id])
                    count.append(id)
            features = data[name][idx, index, :, :]

            # extract corresponding mean and std
            means = np.array(data_dict["mean"])[count]
            stds = np.array(data_dict["std"])[count]

        else:  # all features retrieved and standardized for other datasets
            features = data[name][idx, self.dataset[name]["index"], :, :]
            means = np.array(data_dict["mean"])
            stds = np.array(data_dict["std"])

        # normalization
        return (features - means.T[:, None, None]) / stds.T[:, None, None]

    def revert_feature_standardization(
        self, features: np.array, means: list, stds: list
    ):
        """
        Reverts standardisation for plotting batched image data in tensorboard log

        Parameters
        ----------
        features: np.array
            Array of single samples with single / multiple variables to be transformed
        means: list
            Means of all samples for each feature
        stds: list
            Standard deviations of all samples for each feature

        Returns
        -------
        features: np.array
            Features, same shape as input features, undergone reverted standardization
        """
        for idx, (mean, std) in enumerate(zip(means, stds)):
            features[:, idx, :, :] = (features[:, idx, :, :] * std) + mean

        return features

    def get_one_hot_encoding(self, worldcover_array: np.array):
        """
        Returns an one-hot encoded array with one channel for each class

        Parameters
        ----------
        worldcover_array: np.array
            Land use cover array of single sample

        Returns
        -------
        out: np.array
            Array contains same inforation but one hot encoded
        """
        out = np.zeros(
            (
                len(self.worldcover["class"]),
                worldcover_array.shape[0],
                worldcover_array.shape[1],
            )
        )
        for idx, wc_cls in enumerate(self.worldcover["class"]):
            out[idx, :, :] = np.where(worldcover_array == wc_cls, 1, 0)

        return out

    def fold_worldcover(self, worldcover_one_hot: np.array):
        """
        Creates a single channel array containing the ESA WorldCover classes from a one-hot encoded array

        Parameters
        ----------
        worldcover_array: np.array
            Land use cover array one hot encoded

        Returns
        -------
        out: np.array
            Array contains same inforation but one hot encoded
        """
        out = np.zeros((worldcover_one_hot.shape[1], worldcover_one_hot.shape[1]))
        for idx, wc_cls in enumerate(self.worldcover["class"]):
            out[worldcover_one_hot[idx, :, :] == 1] = wc_cls

        return out
