import argparse
import warnings


class ConfigParser:
    """
    A class to parse config from command line and validate inputs

    ...

    Methods
    ----------
    validate:
      check if all inputs from command line valid.
    parse_args:
      convert arguments into dict. All args will overwrite default from Config class.
      If config file provided, all other arguments will be dismissed.

    """

    def __init__(self):
        pass

    def validate(self, args_dict: dict):
        """
        check validity of input arguments and display user warnings
        """
        # get all config items as dict
        valid_args = {k: v for k, v in args_dict.items() if v is not None}
        # in case of redundant params
        if "config" in valid_args and len(valid_args) > 1:
            warnings.warn(
                "With config file path provided, all other options will be dismissed."
            )
        # reminder for default
        if len(valid_args) == 0:
            warnings.warn(
                "No config file and arguments provided. Model will be run based on all default options."
            )
        return valid_args

    def parse_args(self):
        """
        overwrite default arguments with user inputs
        """
        parser = argparse.ArgumentParser(prog="AGBD Estimation Training and Evaluation")
        parser.add_argument(
            "--config", type=str, required=False, help="Path to config file"
        )
        parser.add_argument(
            "--resume_dir",
            type=str,
            required=False,
            help="Option to resume training from existing training log directory (default False)",
        )
        parser.add_argument(
            "--ds",
            type=str,
            required=False,
            help="Training sample selection, 87k for small dataset, 470k for large dataset, 2e6 for complete dataset (default 87k)",
        )
        parser.add_argument(
            "--mode",
            type=str,
            required=False,
            help="Approach short cut to run the model train (default default). Priority over other options. Use default to enable only customized options.\n\
                            'baseline': turn off all optional operations (except resampling for testing set is on)\n\
                            'baseline_vi': turn off all optional operations except vegetation indices\n\
                            'baseline_rgb': turn off all optional operations and keep only RGB bands\n\
                            'vi': turn on vegetation indices and use defalt for other options\n\
                            'default': predefined optimized combination of options\n",
        )
        parser.add_argument(
            "--log",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to do logging (default True)",
        )
        parser.add_argument(
            "--worldcover",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to turn on world cover layer (default True)",
        )
        parser.add_argument(
            "--elevation",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to turn on elevation from dem as input (default True)",
        )
        parser.add_argument(
            "--slope",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to turn on slope from dem as input (default False)",
        )
        parser.add_argument(
            "--xy",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use lat long as input (default False)",
        )
        parser.add_argument(
            "--loss",
            type=str,
            required=False,
            help="loss function to use 'MaskedMSELoss', 'MaskedRMSELoss', or 'MaskedGaussianNLLLoss'",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            help="batch size for cnn model training (default 32)",
        )
        parser.add_argument(
            "--epochs", type=int, required=False, help="epochs to run (default 100000)"
        )
        parser.add_argument(
            "--debug",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="debug mode to run single sample (default False)",
        )
        parser.add_argument(
            "--transform_mode",
            type=str,
            required=False,
            help="mode for data augmentation 'default' (default), 'crop_only', 'validation', 'test'",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            help="hyperparameter used in the training of neural networks (default 1e-3)",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            required=False,
            help="hyperparameter used in the training of neural networks (default 1e-3)",
        )
        parser.add_argument(
            "--deterministic",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="deterministic behavior to allow result reproduction, False to save training time (default True)",
        )
        parser.add_argument(
            "--save_top_k",
            type=int,
            required=False,
            help="best epoch checkpoints to save, eg. 1 to save the best only, -1, to save all checkpoints. Default 5.",
        )
        parser.add_argument(
            "--checkpoint_interval",
            type=int,
            required=False,
            help="every n epochs to save checkpoint file for resume and evaluation (default 1)",
        )
        parser.add_argument(
            "--early_stop",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to stop training early when validation loss does not decrease further (default True)",
        )
        parser.add_argument(
            "--production",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="run for final production to switch off all experimental options (default False)",
        )
        parser.add_argument(
            "--crop",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="whether to crop image to predefined size (default True)",
        )
        parser.add_argument(
            "--original_size",
            type=int,
            required=False,
            help="image size in both x- and y- directions (default 32)",
        )
        parser.add_argument(
            "--crop_size",
            type=int,
            required=False,
            help="size after cropping, ignored with crop = False (default 16)",
        )
        parser.add_argument(
            "--resample_scale",
            type=int,
            required=False,
            help="sample size/ population to resample (default 2), set 0 to return resampling off",
        )
        parser.add_argument(
            "--log_interval",
            type=int,
            required=False,
            help="epoch intervals for logging (default 50)",
        )
        parser.add_argument(
            "--s2_b02",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b03",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b04",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b05",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b06",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b07",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b08",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b8a",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b09",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b11",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--s2_b12",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use individual Sentinel-2 band (default True)",
        )
        parser.add_argument(
            "--ndvi",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to compute vegetation indices (default False)",
        )
        parser.add_argument(
            "--lai",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to compute vegetation indices (default False)",
        )
        parser.add_argument(
            "--ndwi",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to compute vegetation indices (default False)",
        )
        parser.add_argument(
            "--model_remark",
            type=str,
            required=False,
            help="name to be put on checkpoint folder to identify key characteristics of the model settings (default 'model')",
        )
        parser.add_argument(
            "--redraw_thres",
            type=int,
            required=False,
            help="upper limit for same sample to be drawn in training and validation (default 5)",
        )
        parser.add_argument(
            "--location_bootstrap",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use location for sample bootstrapping (default True)",
        )
        parser.add_argument(
            "--agb_bootstrap",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use biomass value for sample bootstrapping (default True)",
        )
        parser.add_argument(
            "--landuse_bootstrap",
            type=bool,
            action=argparse.BooleanOptionalAction,
            required=False,
            help="option to use land use class within grid for sample bootstrapping (default True)",
        )

        # parse all inputs
        args, _ = parser.parse_known_args()
        args_dict = vars(args)
        # removed unused arguments to take python class default values
        valid_dict = self.validate(args_dict)

        return valid_dict
