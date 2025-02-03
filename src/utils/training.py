# training.py

import random
import numpy as np

from skimage import measure

from torch import Tensor
from torch import optim
import torch.nn as nn

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers

import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors

from utils.losses import MaskedMSELoss, MaskedRMSELoss, MaskedGaussianNLLLoss
from utils.scheduler import CosineAnnealingWarmUpRestarts

import gc
import logging

log = logging.getLogger("lightning.pytorch.core")


# define the LightningModule
class BiomassModel(pl.LightningModule):
    def __init__(self, model, cfg: dict, ds):
        super().__init__()

        # Save Params in Checkpoint File
        # https://lightning.ai/forums/t/hparams-missing-not-saved-in-checkpoints-self-save-hyperparameters-was-called/2098/5
        self.save_hyperparameters(
            logger=False, ignore=["model"]
        )  # save params to checkpoint for prediction and resume training
        self.model = model  # cnn model object
        self.cfg = cfg  # config dict
        self.running_idx_train = 0  # Used for logging
        self.running_idx_val = 0  # Used for logging
        self.ds = ds  # biomass dataset object
        self.loss = self.init_loss(
            cfg["training"]["loss"]
        )  # model loss computed from predefined approach

    def init_loss(self, loss: str, mask: bool = True, no_data: int = -9999):
        """
        Select loss function from selection

        Parameters
        ----------
        loss : str
            One of "MaskedMSELoss", "MaskedRMSELoss", "MaskedGaussianNLLLoss".

        Returns
        -------
        loss function object
            Loss class object corresponding to the selected approach

        """
        if loss == "MaskedMSELoss":
            return MaskedMSELoss(mask=mask, nodata=no_data)
        elif loss == "MaskedRMSELoss":
            return MaskedRMSELoss(mask=mask, nodata=no_data)
        elif loss == "MaskedGaussianNLLLoss":
            return MaskedGaussianNLLLoss(mask=mask, nodata=no_data)  # , neg_penalty=1)
        else:
            raise ValueError("Unknown loss function")

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Activate the training loop from batch samples

        Parameters
        ----------
        batch: tuple
            Single batch of samples including features and labels
        batch_idx: int
            Identity of batch passed

        Returns
        -------
        loss: float
            Computed single loss value from all samples passed

        """
        self.running_idx_train += 1
        # training_step defines the train loop.
        # it is independent of forward step
        x, y = batch  # features and labels
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging loss to TensorBoard (if installed) by default
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        if (
            self.running_idx_train
            % self.cfg["logging"]["img_log_interval_train_batches"]
            == 0
        ):
            # Log random sample to tensor (RGB, AGBD estimate and true AGBD)
            self.log_tb_images(x, y, y_hat, mode="train", idx=self.running_idx_train)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Activate the validation loop from batch samples and record loss in log

        Parameters
        ----------
        batch: tuple
            Single batch of samples including features and labels
        batch_idx: int
            Identity of batch passed

        Returns
        -------
        None

        """
        self.running_idx_val += 1

        # get loss for individual batch
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)

        # log performance
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if (
            self.running_idx_val % self.cfg["logging"]["img_log_interval_val_batches"]
            == 0
        ):
            # Log random sample to tensor (RGB, AGBD estimate and true AGBD)
            self.log_tb_images(x, y, y_hat, mode="val", idx=self.running_idx_val)

    def test_step(self, batch: tuple, batch_idx: int):
        """
        Activate the testing step from batch samples and record loss in log

        Parameters
        ----------
        batch: tuple
            Single batch of samples including features and labels
        batch_idx: int
            Identity of batch passed

        Returns
        -------
        None

        """
        # get loss for individual batch
        x, y = batch
        y_hat = self.model(x)
        rmse = MaskedRMSELoss(mask=True, nodata=-9999)
        test_loss = rmse(y_hat, y)

        # log performance
        self.log(
            "test_rmse",
            test_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        """
        Set up predefined optimizer with learning rate scheduler based on config values for CNN model
        https://github.com/Lightning-AI/lightning/issues/3795
        https://lightning.ai/docs/pytorch/latest/common/optimization.html#id5

        Parameters
        ----------
        None

        Returns
        -------
        A dict of optimizer and scheduler

        """
        # adam optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.cfg["training"]["learning_rate"],
            weight_decay=self.cfg["training"]["weight_decay"],
        )

        # auto adapting learning rate set up
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer)

        return [optimizer], [lr_scheduler]

    def log_tb_images(
        self, x: Tensor, y: Tensor, y_hat: Tensor, mode: str, idx: int
    ) -> None:
        """
        Pass predefined figures computed from training samples to Tensorboard log dashboard

        Parameters
        ----------
        x: Feature tensor
        y: Target tensor
        y_hat: Prediction tensot
        cfg: Config used
        mode: "train" or "val"
        idx: Running training index

        Returns
        -------
        None

        """
        x = x.detach().cpu()
        y = y.detach().cpu()
        y_hat = y_hat.detach().cpu()

        # set up tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")

        # set up RGB image for logging
        rnd_idx = random.choice(range(x.shape[0]))  # get random image

        r_idx = self.cfg["training"]["features"]["channel_names"].index("Sentinel2_B04")
        g_idx = self.cfg["training"]["features"]["channel_names"].index("Sentinel2_B03")
        b_idx = self.cfg["training"]["features"]["channel_names"].index("Sentinel2_B02")
        # r_idx = self.cfg["training"]["features"]["sentinel2"]["b04"]["index"] - 1
        # g_idx = self.cfg["training"]["features"]["sentinel2"]["b03"]["index"] - 1
        # b_idx = self.cfg["training"]["features"]["sentinel2"]["b02"]["index"] - 1

        # if self.cfg["training"]["features"]["xy"]["use"]:
        #     r_idx += 3
        #     g_idx += 3
        #     b_idx += 3

        if self.cfg["logging"]["log_rgb"]:
            features_denormalized = self.ds.revert_feature_standardization(
                x.numpy(), means=self.ds.means, stds=self.ds.stds
            )
            r = features_denormalized[
                rnd_idx,
                r_idx,
                :,
                :,
            ]
            g = features_denormalized[
                rnd_idx,
                g_idx,
                :,
                :,
            ]
            b = features_denormalized[
                rnd_idx,
                b_idx,
                :,
                :,
            ]
            rgb = np.stack((r, g, b), axis=2)
        else:
            rgb = None
        if self.cfg["logging"]["log_worldcover"]:
            worldcover = x[rnd_idx, self.ds.worldcover["index_start"] :, :, :].numpy()
            worldcover = self.ds.fold_worldcover(worldcover)
        else:
            worldcover = None

        # get biomass grond truth label
        agbd_estimate = y_hat[rnd_idx, 0, :, :].numpy()
        agbd_truth = y[rnd_idx, 0, :, :].numpy()

        figsize = (5, 5)

        # retrieve single agb value from image
        agbd_val = str(
            round(agbd_truth.max(), 2)
        )  # For some reason I currenty don't without casting to string the rounding doesn't work
        agbd_estimate_mean = agbd_estimate[agbd_truth > 0].mean()
        agbd_estimate_mean = str(
            round(agbd_estimate_mean, 2)
        )  # For some reason I currenty don't know without casting to string the rounding doesn't work
        amin = float(agbd_val) - 20  # Mg/ha
        amax = float(agbd_val) + 20  # Mg/ha
        contour_linewidth = 4

        if amin < 0:
            amin = 0

        # get GEDI footprint
        contours = measure.find_contours(agbd_truth, level=0)

        # plot variance
        if self.cfg["training"]["output_variance"]:
            agbd_variance = y_hat[rnd_idx, 1, :, :].numpy()
            agbd_variance_mean = str(
                np.round(agbd_variance[agbd_truth > 0].mean(), decimals=2)
            )
            f2a, axs = plt.subplots(figsize=figsize)
            im = axs.imshow(
                agbd_variance, vmin=0, vmax=200, cmap="inferno", aspect="auto"
            )
            axs.set_title(
                f"{idx} - AGBD Estimate Variance at GT: {agbd_variance_mean} Mg/ha"
            )
            axs.axis("off")
            for contour in contours:
                axs.plot(
                    contour[:, 1], contour[:, 0], "white", linewidth=contour_linewidth
                )
                axs.plot(
                    contour[:, 1], contour[:, 0], "k--", linewidth=contour_linewidth - 1
                )
            plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

        # plot biomass with contour
        f1, axs = plt.subplots(figsize=figsize)
        im = axs.imshow(agbd_truth, vmin=amin, vmax=amax, cmap="inferno", aspect="auto")
        plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
        axs.set_title(f"{idx} - AGBD: {agbd_val} Mg/ha")

        for contour in contours:
            axs.plot(contour[:, 1], contour[:, 0], "white", linewidth=contour_linewidth)
            axs.plot(
                contour[:, 1], contour[:, 0], "k--", linewidth=contour_linewidth - 1
            )

        axs.axis("off")

        # plot agb prediction with contour
        f2, axs = plt.subplots(figsize=figsize)
        im = axs.imshow(
            agbd_estimate, vmin=amin, vmax=amax, cmap="inferno", aspect="auto"
        )
        axs.set_title(f"{idx} - AGBD Estimate Mean at GT: {agbd_estimate_mean} Mg/ha")
        axs.axis("off")
        for contour in contours:
            axs.plot(contour[:, 1], contour[:, 0], "white", linewidth=contour_linewidth)
            axs.plot(
                contour[:, 1], contour[:, 0], "k--", linewidth=contour_linewidth - 1
            )
        plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

        # set up Sentinel-2 plot
        f3, axs = plt.subplots(figsize=figsize)

        scale_factor = np.percentile(rgb, 90)
        amin = 0
        amax = 1.0

        # Scale values
        if rgb is not None:
            im = rgb = np.clip((rgb / scale_factor), a_min=amin, a_max=amax)
            axs.imshow(rgb, vmin=amin, vmax=amax, aspect="auto")
            for contour in contours:
                axs.plot(
                    contour[:, 1], contour[:, 0], "white", linewidth=contour_linewidth
                )
                axs.plot(
                    contour[:, 1], contour[:, 0], "k--", linewidth=contour_linewidth - 1
                )
            axs.set_title(f"{idx} - Sentinel-2 RGB")
            axs.axis("off")
        else:
            axs.title("no RGB data")

        # visualize land use from WorldCover
        if self.ds.worldcover["feature"]["use"]:
            f4, axs = plt.subplots(figsize=figsize)
            colors = [
                "green",
                "orange",
                "yellow",
                "violet",
                "red",
                "gray",
                "whitesmoke",
                "blue",
                "lightseagreen",
                "springgreen",
                "navajowhite",
            ]
            cmap, norm = from_levels_and_colors(
                [0] + [val + 0.1 for val in self.ds.worldcover["class"]], colors
            )
            im = axs.imshow(
                worldcover, cmap=cmap, norm=norm, interpolation="none", aspect="auto"
            )
            for contour in contours:
                axs.plot(
                    contour[:, 1], contour[:, 0], "white", linewidth=contour_linewidth
                )
                axs.plot(
                    contour[:, 1], contour[:, 0], "k--", linewidth=contour_linewidth - 1
                )
            axs.set_title(f"{idx} - WorldCover")
            axs.axis("off")
            cbar = plt.colorbar(
                im,
                ax=axs,
                fraction=0.046,
                pad=0.04,
                ticks=[5, 15, 25, 35, 45, 55, 65, 75, 85, 92.5, 97.5],
            )
            cbar.ax.set_yticklabels(
                [
                    "Tree Cover",
                    "Shrubland",
                    "Grassland",
                    "Cropland",
                    "Built-Up",
                    "Sparse",
                    "Snow/Ice",
                    "Water",
                    "Wetland",
                    "Mangroves",
                    "Moos",
                ]
            )

        plt.tight_layout()

        # pass figures
        tb_logger.add_figure(f"{mode}/image/AGBD_True", f1, idx)
        tb_logger.add_figure(f"{mode}/image/AGBD_Pred", f2, idx)
        tb_logger.add_figure(f"{mode}/image/Sentinel-2", f3, idx)

        # close figures
        f1.clf()
        f2.clf()
        f3.clf()

        plt.close(f1)
        plt.close(f2)
        plt.close(f3)

        # add world cover figure
        if self.ds.worldcover["feature"]["use"]:
            tb_logger.add_figure(f"{mode}/image/WorldCover", f4, idx)
            f4.clf()
            plt.close(f4)
            del worldcover, f4

        # add variance plot figure
        if self.cfg["training"]["output_variance"]:
            tb_logger.add_figure(f"{mode}/image/Variance", f2a, idx)
            f2a.clf()
            plt.close(f2a)
            del agbd_variance, f2a

        # clean variables
        del f1, f2, f3, axs, im, rgb, agbd_estimate, agbd_truth

        gc.collect()
