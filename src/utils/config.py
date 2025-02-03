# config.py

import os
from utils.processing_utils import load_config
import numpy as np


class Config:
    """
    A class to construct config dictionary for cnn model train, which automatically
    manages all option dependencies and allow config short cuts e.g. rgb.

    ...

    Attributes
    ----------
    ds: str
        training sample selection, 87k for small dataset,
        470k for large dataset, 2e6 for complete dataset (default "87k")
    mode: str
        approach short cut to run the model train (default "default")
        priority over other options. Use "default" to enable only customized options.
        "baseline": turn off all optional operations
        "baseline_vi": turn off all optional operations except vegetation indices
        "baseline_rgb": turn off all optional operations and keep only RGB bands
        "vi": turn on vegetation indices and use defalt for other options
        "default": predefined optimized combination of options
    log: bool
        option to do logging (default True)
    worldcover: bool
        option to turn on world cover layer (default True)
    elevation: bool
        option to turn on elevation from dem as input (default True)
    slope: bool
        option to turn on slope from dem as input (default False)
    xy: bool
        option to use lat long as input (default False)
    loss: str
        loss function to use "MaskedMSELoss", "MaskedRMSELoss", or "MaskedGaussianNLLLoss"
        (default "MaskedGaussianNLLLoss")
    batch_size: int
        batch size for cnn model training (default 32)
    epochs: int
        epochs to run (default 100000)
    debug: bool
        debug mode to run single sample (default False)
    transform_mode: str
        mode for data augmentation "default" (default), "crop_only", "validation", "test"
    learning_rate: float
        hyperparameter used in the training of neural networks (default 1e-3)
    weight_decay: float
        hyperparameter used in the training of neural networks (default 1e-3)
    deterministic: bool
        deterministic behavior to allow result reproduction, False to save training time
        (default True)
    save_top_k: int
        best epoch checkpoints to save, eg. 1 to save the best only (default -1, to save all checkpoints)
    checkpoint_interval: int
        every n epochs to save checkpoint file for resume and evaluation (default 5)
    production: bool
        run for final production to switch off all experimental options (default False)
    crop: bool
        whether to crop image to predefined size (default True)
    original_size: int
        image size in both x- and y- directions (default 32)
    crop_size: int
        size after cropping, ignored with crop = False (default 16)
    resample_scale: int
        sample size/ population to resample (default 2)
    early_stop: bool
        option to stop training early when validation loss does not decrease further (default True)
    log_interval: int
        epoch intervals for logging (default 50)
    s2_b02: bool
        option to use individual Sentinel-2 band (default True)
    s2_b03: bool
        option to use individual Sentinel-2 band (default True)
    s2_b04: bool
        option to use individual Sentinel-2 band (default True)
    s2_b05: bool
        option to use individual Sentinel-2 band (default True)
    s2_b06: bool
        option to use individual Sentinel-2 band (default True)
    s2_b07: bool
        option to use individual Sentinel-2 band (default True)
    s2_b08: bool
        option to use individual Sentinel-2 band (default True)
    s2_b8a: bool
        option to use individual Sentinel-2 band (default True)
    s2_b09: bool
        option to use individual Sentinel-2 band (default True)
    s2_b11: bool
        option to use individual Sentinel-2 band (default True)
    s2_b12: bool
        option to use individual Sentinel-2 band (default True)
    ndvi: bool
        option to compute vegetation indices (default False)
    lai: bool
        option to compute vegetation indices (default False)
    ndwi: bool
        option to compute vegetation indices (default False)
    model_remark: str
        name to be put on checkpoint folder to identify key characteristics of the model settings
        (default "model")

    Methods
    -------
    __call__():
        return constructed config dict for model train.
    """

    s2_bands = [
        "b02",
        "b03",
        "b04",
        "b05",
        "b06",
        "b07",
        "b08",
        "b8a",
        "b09",
        "b11",
        "b12",
        "ndvi",
        "lai",
        "ndwi",
    ]
    s2_mean = {
        "87k": [
            468.31,
            725.86,
            801.86,
            1242.93,
            2249.38,
            2628.94,
            2724.25,
            2878.07,
            2884.21,
            2301.29,
            1518.38,
            0,
            0,
            0,
            0,
        ],
        "470k": [
            469.57,
            727.63,
            804.81,
            1245.84,
            2252.07,
            2632.34,
            2727.03,
            2881.53,
            2887.59,
            2306.05,
            1523.08,
            0,
            0,
            0,
            0,
        ],
        "2e6": [
            469.57,
            727.63,
            804.81,
            1245.84,
            2252.07,
            2632.34,
            2727.03,
            2881.53,
            2887.59,
            2306.05,
            1523.08,
            0,
            0,
            0,
            0,
        ],
    }

    s2_std = {
        "87k": [
            377.9,
            514.11,
            846.27,
            819.08,
            750.03,
            850.2,
            880.03,
            867.91,
            839.2,
            1165.31,
            1192.22,
            1,
            1,
            1,
            1,
        ],
        "470k": [
            377.74,
            513.69,
            846.27,
            818.66,
            749.37,
            850.79,
            880.45,
            868.74,
            839.03,
            1165.54,
            1193.6,
            1,
            1,
            1,
            1,
        ],
        "2e6": [
            377.74,
            513.69,
            846.27,
            818.66,
            749.37,
            850.79,
            880.45,
            868.74,
            839.03,
            1165.54,
            1193.6,
            1,
            1,
            1,
            1,
        ],
    }

    dem_mean = {"87k": [466.32, 4.96], "470k": [465.44, 4.96], "2e6": [465.44, 4.96]}

    dem_std = {"87k": [513.94, 5.01], "470k": [513.02, 5.0], "2e6": [513.02, 5.0]}

    def __init__(
        self,
        ds="87k",
        mode="default",
        log=True,
        worldcover=True,
        elevation=True,
        slope=False,
        xy=False,
        loss="MaskedGaussianNLLLoss",
        batch_size=32,
        epochs=100000,
        debug=False,
        transform_mode="default",
        learning_rate=1e-3,
        weight_decay=1e-3,
        deterministic=True,
        save_top_k=5,  # get all epoch checkpoints
        checkpoint_interval=1,  # every_n_epochs
        production=False,
        crop=True,
        original_size=32,
        crop_size=16,
        resample_scale=2,
        early_stop=True,
        log_interval=50,
        s2_b02=True,
        s2_b03=True,
        s2_b04=True,
        s2_b05=True,
        s2_b06=True,
        s2_b07=True,
        s2_b08=True,
        s2_b8a=True,
        s2_b09=True,
        s2_b11=True,
        s2_b12=True,
        ndvi=False,
        lai=False,
        ndwi=False,
        model_remark=None,
        redraw_thres=5,
        location_bootstrap=True,
        agb_bootstrap=True,
        landuse_bootstrap=True,
        resume_dir=False,
    ):
        if resume_dir:
            self.resume_dir = resume_dir
            pass

        else:
            self.resume_dir = False

            # adjustment from shortcut params
            if not model_remark:
                model_remark = mode

            if debug:
                resample_scale = None

            if production:  # production mode
                save_top_k = 1
                checkpoint_interval = None
                ds = "2e6"
                debug = False
                deterministic = True

            if not (location_bootstrap or agb_bootstrap or landuse_bootstrap):
                resample_scale = None

            # auto params set for mode selection
            if mode == "baseline":  # turn off all optional operations
                worldcover, elevation, slope = False, False, False
                ndvi, lai, ndwi = False, False, False
                transform_mode = "crop_only"
                resample_scale = None
                xy = False

            elif mode == "baseline_rgb":  # get baseline with 3 color channels only
                s2_b05, s2_b06, s2_b07, s2_b08, s2_b8a, s2_b09, s2_b11, s2_b12 = (
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                )
                worldcover, elevation, slope = False, False, False
                ndvi, lai, ndwi = False, False, False
                transform_mode = "crop_only"
                resample_scale = None
                xy = False

            elif mode == "baseline_vi":  # baseline with vegetation indices
                worldcover, elevation, slope = False, False, False
                ndvi, lai, ndwi = True, True, True
                transform_mode = "crop_only"
                resample_scale = None

            elif mode == "rgb":  # turn on rgb and discard all other channels
                s2_b02, s2_b03, s2_b04 = True, True, True
                s2_b05, s2_b06, s2_b07, s2_b08, s2_b8a, s2_b09, s2_b11, s2_b12 = (
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                )
                ndvi, lai, ndwi = False, False, False

            elif mode == "vi":  # compute vegetation indices
                s2_b02, s2_b03, s2_b04, s2_b08 = True, True, True, True
                ndvi, lai, ndwi = True, True, True

            elif mode == "default":  # keep default settings
                transform_mode = "default"

            # construct config dict
            self.config_dict = {"dataset": {}, "logging": {}, "training": {}}
            self.config_dict["production"] = production  # production flag

            self.config_dict["training"]["lightning"] = {}
            self.config_dict["training"]["features"] = {}
            self.home = os.path.expanduser("~")  # user home dir
            self.mode = mode

            self.worldcover = worldcover
            self.elevation = elevation
            self.slope = slope
            self.xy = xy
            self.data = ds
            self.s2_use = [
                s2_b02,
                s2_b03,
                s2_b04,
                s2_b05,
                s2_b06,
                s2_b07,
                s2_b08,
                s2_b8a,
                s2_b09,
                s2_b11,
                s2_b12,
                ndvi,
                lai,
                ndwi,
            ]

            self.dataset = self.config_dict["dataset"]
            self.logging = self.config_dict["logging"]
            self.training = self.config_dict["training"]

            # compute channel counts
            self.count_channels(
                elevation=elevation, slope=slope, worldcover=worldcover, xy=xy
            )

            # fill params in dict
            ################
            ### dataset  ###
            ################
            self.set_dataset(ds)

            ################
            ### logging  ###
            ################
            self.set_logging(log, worldcover=worldcover, rgb=(s2_b02 & s2_b03 & s2_b04))

            ################
            ### training ###
            ################
            self.set_training(
                debug=debug,
                batch_size=batch_size,
                epochs=epochs,
                loss=loss,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                deterministic=deterministic,
                crop=crop,
                original_size=original_size,
                crop_size=crop_size,
                in_channels=self.in_channels,
                resample_scale=resample_scale,
                transform_mode=transform_mode,
                redraw_thres=redraw_thres,
                location_bootstrap=location_bootstrap,
                agb_bootstrap=agb_bootstrap,
                landuse_bootstrap=landuse_bootstrap,
            )

            ##################
            ### lightning  ###
            ##################
            self.set_lightning(
                log_interval=log_interval,
                save_top_k=save_top_k,
                checkpoint_interval=checkpoint_interval,
                model_remark=model_remark,
                early_stop=early_stop,
            )

            ################
            ### features ###
            ################
            self.set_features(
                worldcover=worldcover, elevation=elevation, slope=slope, xy=xy
            )

    def count_channels(self, elevation: bool, slope: bool, worldcover: bool, xy: bool):
        """compute channel count for cnn input"""

        self.channel_names = []

        count = 0
        count += sum(self.s2_use)

        if xy:
            count += 4
            self.channel_names.extend(["lat_sin", "lat_cos", "lon_sin", "lon_cos"])

        s2_names = np.array(
            [
                "Sentinel2_B02",
                "Sentinel2_B03",
                "Sentinel2_B04",
                "Sentinel2_B05",
                "Sentinel2_B06",
                "Sentinel2_B07",
                "Sentinel2_B08",
                "Sentinel2_B8a",
                "Sentinel2_B09",
                "Sentinel2_B11",
                "Sentinel2_B12",
                "NDVI",
                "LAI",
                "NDWI",
            ]
        )
        self.channel_names.extend(
            s2_names[np.array(self.s2_use)]  # multiple channel indices
        )
        print(self.channel_names)
        if elevation:
            count += 1
            self.channel_names.extend(["Elevation"])
        if slope:
            count += 1
            self.channel_names.extend(["Slope"])
        if worldcover:
            count += 11
            worldcover_names = [
                "World Cover Tree cover (10)",
                "World Cover Shrubland (20)",
                "World Cover Grassland (30)",
                "World Cover Cropland (40)",
                "World Cover Built-up (50)",
                "World Cover Bare / sparse vegetation (60)",
                "World Cover Snow and ice (70)",
                "World Cover Permanent water bodies (80)",
                "World Cover Herbaceous wetland (90)",
                "World Cover Mangroves (95)",
                "World Cover Moss and lichen (100)",
            ]
            self.channel_names.extend(worldcover_names)

        assert len(self.channel_names) == count, "Error in counting channel names!"
        self.training["features"]["channel_names"] = [
            f"{i}" for i in self.channel_names
        ]  # np.array(self.channel_names)

        self.in_channels = count

    def set_features(self, worldcover: bool, elevation: bool, slope: bool, xy: bool):
        """set up feature statistics"""
        d = self.training["features"]
        d["sentinel2"], d["copdem"], d["worldcover"], d["xy"] = {}, {}, {}, {}

        for i, band in enumerate(self.s2_bands):
            d["sentinel2"][band] = {}
            d["sentinel2"][band]["use"] = self.s2_use[i]
            d["sentinel2"][band]["index"] = i + 1
            d["sentinel2"][band]["mean"] = self.s2_mean[self.data][i]
            d["sentinel2"][band]["std"] = self.s2_std[self.data][i]

        d["copdem"]["altitude"] = {}
        d["copdem"]["altitude"]["use"] = elevation
        d["copdem"]["altitude"]["index"] = 0
        d["copdem"]["altitude"]["mean"] = self.dem_mean[self.data][0]
        d["copdem"]["altitude"]["std"] = self.dem_std[self.data][0]

        d["copdem"]["slope"] = {}
        d["copdem"]["slope"]["use"] = slope
        d["copdem"]["slope"]["index"] = 0
        d["copdem"]["slope"]["mean"] = self.dem_mean[self.data][1]
        d["copdem"]["slope"]["std"] = self.dem_std[self.data][1]

        d["worldcover"]["use"] = worldcover
        d["xy"]["use"] = xy

    def set_lightning(
        self,
        log_interval: int,
        save_top_k: int,
        checkpoint_interval: int,
        model_remark: str,
        early_stop: bool,
    ):
        """set up model logging and checkpoint options"""
        d = self.training["lightning"]
        d["log_every_n_batches"] = log_interval
        d["save_top_k"] = save_top_k
        d["checkpoint_interval"] = checkpoint_interval
        d["model_remark"] = model_remark
        d["early_stop"] = early_stop

    def set_training(
        self,
        debug: bool,
        batch_size: int,
        epochs: int,
        loss: str,
        learning_rate: float,
        weight_decay: float,
        deterministic: bool,
        crop: bool,
        original_size: int,
        crop_size: int,
        in_channels: int,
        resample_scale: float,
        transform_mode: str,
        redraw_thres: int,
        location_bootstrap: bool,
        agb_bootstrap: bool,
        landuse_bootstrap: bool,
    ):
        """set up training params"""
        d = self.training
        d["single_sample_debug"] = debug
        d["single_sample_debug_idx"] = 0
        d["fast_dev_run"] = debug
        d["batch_size"] = batch_size
        d["epochs"] = epochs
        d["loss"] = loss
        d["output_variance"] = True if loss == "MaskedGaussianNLLLoss" else False
        d["learning_rate"] = learning_rate
        d["weight_decay"] = weight_decay
        d["seed"] = 1  # set random seed
        d["cuda_deterministic_mode"] = deterministic
        d["use_random_crop"] = crop if not debug else False
        d["original_size"] = original_size  # pixels
        d["crop_size"] = (
            crop_size if d["use_random_crop"] else d["original_size"]
        )  # pixels
        d["in_channels"] = in_channels
        d["resample_scale"] = resample_scale
        d["transform_mode"] = transform_mode
        d["redraw_thres"] = redraw_thres
        d["location_bootstrap"] = location_bootstrap
        d["agb_bootstrap"] = agb_bootstrap
        d["landuse_bootstrap"] = landuse_bootstrap

    def set_logging(self, log: bool, worldcover: bool, rgb: bool):
        """set up logging params"""
        d = self.logging
        d["logdir"] = os.path.join(self.home, "biomass-estimation", "training_logs")
        d["img_log_interval_train_batches"] = 500
        d["img_log_interval_val_batches"] = 100
        d["log_rgb"] = (
            True if (log and rgb) else False
        )  # For this, the corresponding bands must be activated
        d["log_worldcover"] = (
            True if (log and worldcover) else False
        )  # For this, the worldcover must be used as feature

    def set_dataset(self, ds: str):
        """set up dataset used for model training"""
        # dataset options
        paths = {
            "87k": "biomass_dataset_train_87k.h5",
            "470k": "biomass_dataset_train_470k.h5",
            "2e6": "biomass_dataset_train_2e6.h5",
        }
        self.dataset["path_train"] = os.path.join(
            self.home, "biomass-estimation", "data", paths[ds]
        )

    def __call__(self):
        # read previous config in case of training resume
        if self.resume_dir:
            cfg_path = os.path.join(self.resume_dir, "used_config.toml")
            cfg = load_config(cfg_path)
            cfg["resume_dir"] = self.resume_dir

            # add flag to log dir for model resume
            if cfg["training"]["lightning"]["model_remark"].endswith("_model_resumed"):
                pass
            else:
                cfg["training"]["lightning"][
                    "model_remark"
                ] += "_model_resumed"  # mark in folder name

            return cfg
        # training from stratch
        else:
            return self.config_dict
