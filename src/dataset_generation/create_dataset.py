"This script creates a dataset from a set set of GEDI L4A 2.1 data files. The dataset is saved as a HDF5 file." ""
import os
import argparse
import datetime

import random
import numpy as np

import sys


from dataset_generation_utils import (
    create_combined_csv,
    plot_gedi_footprint_distribution,
    create_subset,
    create_hdf_dataset,
    fuse_hdf5_datasets,
)

import logging
import logging.handlers
import multiprocessing


LOGDIR = "/home/max/repos/biomass-estimation/src/dataset_generation/logs"
os.makedirs(LOGDIR, exist_ok=True)


def logger_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages.
    root.setLevel(logging.INFO)


# https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
def listener_configurer():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    filename = os.path.join(
        LOGDIR,
        f"{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')}.log",
    )
    h = logging.FileHandler(filename)
    # h2 = logging.StreamHandler(sys.stdout)
    f = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )

    h.setFormatter(f)
    # h2.setFormatter(f)
    root.addHandler(h)
    # root.addHandler(h2)


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if (
                record is None
            ):  # We send this as a sentinel to tell the listener to quit.
                print("Exiting listener process")
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a dataset from a set of GEDI L4A 2.1 data files."
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output directory where the dataset will be saved.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input directory containing the GEDI L4A 2.1 data files (HDF5).",
    )
    parser.add_argument(
        "--plot-distribution",
        action="store_true",
        help="Plot the distribution of the dataset.",
    )
    parser.add_argument(
        "--path-aggregated-csv-train",
        type=str,
        help="Path to the aggregated CSV file containing the GEDI L4A 2.1 data footprints for the training set. If not provided this file will be created by parsing the HDF5 files.",
    )
    parser.add_argument(
        "--path-aggregated-csv-test",
        type=str,
        help="Path to the aggregated CSV file containing the GEDI L4A 2.1 data footprints for the test set. If not provided this file will be created by parsing the HDF5 files.",
    )
    parser.add_argument(
        "--path-subset-csv-train",
        type=str,
        help="Path to CSV file containg already exactly the footprints the dataset shall be created with. If provided, HDF5 to CSV conversion and subsetting are skipped.",
    )
    parser.add_argument(
        "--path-subset-csv-test",
        type=str,
        help="Path to CSV file containg already exactly the footprints the dataset shall be created with. If provided, HDF5 to CSV conversion and subsetting are skipped.",
    )
    parser.add_argument(
        "--max-size-training",
        type=int,
        default=6e6,
        help="Maximum size of the training dataset in samples.",
    )
    parser.add_argument(
        "--max-size-test",
        type=int,
        default=2e6,
        help="Maximum size of the test dataset in samples.",
    )

    parser.add_argument(
        "--fuse-hdf5-datasets",
        type=str,
        help="Exclusive option to combine the HDF5 dataset at the given location (all must follow the same schema). If provided, all other options are ignored.",
    )

    args = parser.parse_args()

    m = multiprocessing.Manager()
    q = m.Queue()

    # Initiate logging
    listener = multiprocessing.Process(
        target=listener_process, args=(q, listener_configurer)
    )
    listener.start()

    logger_configurer(q)
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    random.seed(0)
    np.random.seed(0)

    if args.fuse_hdf5_datasets:
        log.info(f"Combining HDF5 dataset at location {args.fuse_hdf5_datasets}")
        path_dataset = fuse_hdf5_datasets(args.fuse_hdf5_datasets, args.output)
        log.info(f"Combined dataset saved at {path_dataset}")
        q.put_nowait(None)
        listener.join()
        sys.exit(0)

    # Get the list of GEDI files two process
    gedi_source_files = [
        os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".h5")
    ]

    if not (args.path_aggregated_csv_train and args.path_aggregated_csv_test):
        # Create an aggregated CSV file containing the combined data from all provided GEDI HFD5 files
        log.info(
            "Creating aggregated CSV files for train and test set based on GEDI HDF5 files."
        )
        path_aggregated_csv_train, path_aggregated_csv_test = create_combined_csv(
            gedi_source_files, args.output, "combined.csv"
        )
    else:
        log.info("Using provided aggregated CSV files for train and test set.")
        if not (
            os.path.exists(args.path_aggregated_csv_train)
            and os.path.exists(args.path_aggregated_csv_test)
        ):
            raise ValueError("Provided path to aggregated CSV files do not exist.")
        path_aggregated_csv_train = args.path_aggregated_csv_train
        path_aggregated_csv_test = args.path_aggregated_csv_test

    if args.plot_distribution:
        log.info("Plotting spatial and temporal dataset distribution.")

        plot_gedi_footprint_distribution(
            path_aggregated_csv_train,
            args.output,
            "train_set_distribution_all_footprints.png",
        )
        plot_gedi_footprint_distribution(
            path_aggregated_csv_test,
            args.output,
            "test_set_distribution_all_footprint.png",
        )
        log.info(f"Done. Plots can be found in the output directory: {args.output}")
        sys.exit()

    if not (args.path_subset_csv_train and args.path_subset_csv_test):
        log.info("Commencing subset creation since no subset CSV files are provided")
        csv_subset_train = create_subset(
            path_aggregated_csv_train,
            args.output,
            filename_plot="train_distribution_subset.png",
            filename_csv="train_subset.csv",
            dataset_size_max=int(6e6),
        )
        csv_subset_test = create_subset(
            path_aggregated_csv_test,
            args.output,
            filename_plot="test_distribution_subset.png",
            filename_csv="test_subset.csv",
            dataset_size_max=int(2e6),
        )
    else:
        log.info("Using provided subset CSV files")
        csv_subset_train = args.path_subset_csv_train
        csv_subset_test = args.path_subset_csv_test

    log.info(f"Commencing HDF5 dataset generation.")
    create_hdf_dataset(
        csv_subset_train,
        output_dir=args.output,
        num_processors=24,
        batch_size=1000,
        queue=q,
    )

    log.info("Done.")
    q.put_nowait(None)
    listener.join()
