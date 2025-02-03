import datetime
import logging
import multiprocessing as mp
import os
import random
import re
import string

import time

from itertools import repeat
from typing import Callable, Tuple, Union

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import polars as pl
import pystac_client
import rasterio

from pystac.extensions.eo import EOExtension as eo
from rasterio import features, warp, windows
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform

from shapely.geometry import Point, shape

from requests.adapters import HTTPAdapter
from urllib3 import Retry

from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

import sys


def fuse_hdf5_datasets(path_to_hdf5_files: str, output_dir: str):
    """Fuses all HDF5 files in the output_dir into one HDF5 file."""
    datasets = os.listdir(path_to_hdf5_files)
    datasets = [
        os.path.join(path_to_hdf5_files, d) for d in datasets if d.endswith(".h5")
    ]
    filename = os.path.join(output_dir, "fused_dataset.h5")
    with h5py.File(filename, "w") as f:
        length_final_dataset = 0
        cnt_fail = 0
        for dataset in datasets:
            try:
                with h5py.File(dataset, "r") as f2:
                    # Make sure all datasets within a single HDF5 dataset have the same length
                    len_s2 = len(f2["data"]["Sentinel-2 L2A"])
                    for key in f2["data"].keys():
                        assert len(f2["data"][key]) == len_s2
                    length_final_dataset += len_s2
            except OSError:
                continue

        # collect all data
        f.create_group("data")
        idx_start = 0
        for dataset in datasets:
            try:
                with h5py.File(dataset, "r") as f2:
                    len_ds = len(f2["data"]["Sentinel-2 L2A"])
                    idx_end = idx_start + len_ds
                    for key in f2["data"].keys():
                        chunk_shape = f2["data"][key].chunks
                        ds = f["data"].require_dataset(
                            key,
                            (length_final_dataset, *f2["data"][key].shape[1:]),
                            dtype=f2["data"][key].dtype,
                            chunks=chunk_shape,
                        )
                        ds[idx_start:idx_end] = f2["data"][key][:]
                idx_start = idx_end
            except OSError:
                cnt_fail += 1
                continue
    logging.info(
        f"Found {cnt_fail} files that could not be opened out of {len(datasets)} files"
    )
    return filename


def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    f = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    handler.setFormatter(f)
    root.addHandler(handler)
    # send all messages.
    root.setLevel(logging.INFO)


def create_hdf_dataset(
    path_csv: str, output_dir: str, num_processors: int, batch_size: int, queue
) -> str:
    """Finds intersecting Sentinel-2 products based on a Polars DataFrame containing GEDI footprints and creates a HDF5 dataset using mulitprocessing.

    The created datasets are only of length batch_size and are saved in the output_dir. They must be combined afterwards.
    """
    df = pl.read_csv(path_csv, separator=";", try_parse_dates=True)

    # First step: Retrieve a list of Sentinel-2 L2A products from ±10 days around the GEDI acquisiton date

    # Set automatic retries for failed requests
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[404, 502, 503, 504])
    stac_api_io = StacApiIO()
    stac_api_io.session.mount("https://", HTTPAdapter(max_retries=retry))
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_io=stac_api_io,
        modifier=planetary_computer.sign_inplace,
    )

    n_processes = num_processors
    current_batch = []
    output_dir = os.path.join(output_dir, "hdf5_batch_data")
    os.makedirs(output_dir, exist_ok=True)
    num_frames = int(len(df) / batch_size)
    # Order dataframe by shot number to
    frame_idx = 0
    for frame in df.iter_slices(n_rows=batch_size):
        logging.info(f"Processing frame {frame_idx}/{num_frames}")
        if len(os.listdir(output_dir)) > 0:
            batch_sizes = [
                int(f.removesuffix(".h5").split("_")[-2])
                for f in os.listdir(output_dir)
            ]
            if len(set(batch_sizes)) > 1 or list(set(batch_sizes))[0] != batch_size:
                raise ValueError(
                    "Batch sizes of already processed frames do not match the current batch size. Aborting."
                )
        processed_frames = [
            int(f.removesuffix(".h5").split("_")[-1]) for f in os.listdir(output_dir)
        ]
        if frame_idx in processed_frames:
            logging.info(f"Frame {frame_idx} already processed, skipping...")
            frame_idx += 1
            continue
        current_batch.append((frame, frame_idx))
        if len(current_batch) == n_processes:
            procs = []
            for idx, b in enumerate(current_batch):
                p = mp.Process(
                    target=process_batch,
                    args=(b, catalog, output_dir, worker_configurer, queue, batch_size),
                )
                procs.append(p)
                p.start()
                print("starting", p.name)
            TIMEOUT = 90 * 60  # Kill after n seconds in case a prior process got stuck
            start = time.time()
            while time.time() - start <= TIMEOUT:
                if not any(p.is_alive() for p in procs):
                    # All the processes are done, break now.
                    logging.info("All processes finished")
                    break
                time.sleep(1)
            else:
                # We only enter this if we didn't 'break' above.
                logging.warning("timed out, killing all processes")
                for p in procs:
                    p.terminate()
                    p.join()

            current_batch = []
        frame_idx += 1

    return path_csv


def compute_scl_stats(scl_data: np.array) -> Tuple[dict, bool]:
    """Computes the statistics of the Sentinel-2 SCL data at the footprint location and checks if the footprint is valid.

    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview


    Returns:
        Dictionary containnig the statistics for all classes plus a boolean indicating if the footprint is valid.
    """
    classes = {
        0: "NO_DATA",
        1: "SATURATED_OR_DEFECTIVE",
        2: "CAST_SHADOWS",
        3: "CLOUD_SHADOWS",
        4: "VEGETATION",
        5: "NOT_VEGETATED",
        6: "WATER",
        7: "UNCLASSIFIED",
        8: "CLOUD_MEDIUM_PROBABILITY",
        9: "CLOUD_HIGH_PROBABILITY",
        10: "THIN_CIRRUS",
        11: "SNOW or ICE",
    }

    MAX_CLOUD_FRACTION = 0.25
    MAX_CLOUD_SHADOW_FRACTION = 0.25
    MAX_SNOW_FRACTION = 0.25
    MAX_WATER_FRACTION = 0.25
    MAX_INVALID_FRACTION = 0.05

    stats = {}
    cnt_valid_center_pixels = 0
    for cls_id, cls in classes.items():
        # Check if the 16 center pixels are of class vegetation or non-vegetated (location of the GEDI footprint)
        if cls_id in [4, 5]:
            cnt_valid_center_pixels += np.count_nonzero(
                scl_data[14:18, 14:18] == cls_id
            )

        # Compute the relative occurence of the current class
        num_pixels_cls = np.count_nonzero(scl_data == cls_id)
        cls_fraction = np.around(num_pixels_cls / scl_data.size, decimals=2)
        stats[cls] = cls_fraction

    if cnt_valid_center_pixels != 16:
        return None, False

    elif (
        stats["CLOUD_HIGH_PROBABILITY"]
        + stats["CLOUD_MEDIUM_PROBABILITY"]
        + stats["THIN_CIRRUS"]
        > MAX_CLOUD_FRACTION
    ):
        return None, False

    elif stats["CLOUD_SHADOWS"] + stats["CAST_SHADOWS"] > MAX_CLOUD_SHADOW_FRACTION:
        return None, False

    elif stats["SNOW or ICE"] > MAX_SNOW_FRACTION:
        return None, False

    elif stats["WATER"] > MAX_WATER_FRACTION:
        return None, False

    elif (
        stats["NO_DATA"] + stats["UNCLASSIFIED"] + stats["SATURATED_OR_DEFECTIVE"]
        > MAX_INVALID_FRACTION
    ):
        return None, False

    return stats, True


def compute_slope_degrees(dem: np.array, scale_z: float = 1.0) -> Tuple[np.array, bool]:
    """Computes the slope in degrees from a DEM and checks that the slope at the footprint position is below 15 degrees (SHendryk et al., 2022).)"""
    # ChatGPT: Calculate the slope in the x and y directions using gradient
    dz_dx, dz_dy = np.gradient(dem)

    # Compute the slope magnitude
    slope = np.sqrt(dz_dx**2 + dz_dy**2) * scale_z

    # Convert slope to degrees
    slope_degrees = np.degrees(np.arctan(slope))

    if np.where(slope_degrees[14:18, 14:18] > 15, 1, 0).sum() > 0:
        return None, False

    return slope_degrees, True


def retrieve_data_for_footprint(
    catalog: pystac_client.Client,
    output_dir: str,
    footprint: dict,
    sentinel_2_stack_item_id: str,
) -> None:
    """Retrieves all required data for a GEDI footprint.

    Args:
        catalog: PySTAC client
        output_dir: Full path to directory where to store the data
        footprint: Dict containg the footprint information
        sentinel_2_stack_item_id: ID of the Sentinel-2 stack item
    """
    try:
        location = Point(footprint["lon_lowestmode"], footprint["lat_lowestmode"])

        year = footprint["acquisition_time"].year
        sentinel_2_href = get_sentinel2_hrefs(catalog, sentinel_2_stack_item_id.id)
        copdem_hrefs, worldcover_href = get_aux_hrefs_for_location(
            catalog, location, year=year
        )
        if not copdem_hrefs or not worldcover_href:
            raise ValueError("CopDEM/WorldCover")
        # Extract the product names
        s2_product_name = re.search(
            r"\w{3}_MSIL2A_\w{15}_\w{5}_\w{4}_\w{6}_\w{15}",
            [item for item in list(sentinel_2_href.items()) if item[0] == "B02"][0][1],
        ).group()
        copdem_product_name = re.search(
            r"Copernicus_DSM_COG_\w{2}_\w{1}\d{2}_\w{2}_\w{4}_\w{2}_DEM",
            copdem_hrefs["data"],
        ).group()
        worldcover_product_name = re.search(
            r"ESA_WorldCover_10m_\w{4}_\w{4}_\w{7}_Map",
            worldcover_href["map"],
        ).group()
        if not s2_product_name:
            raise ValueError("product name sentinel-2")
        elif not copdem_product_name:
            raise ValueError("product name copdem")
        elif not worldcover_product_name:
            raise ValueError("product name worldcover")

        # Read image data at footprint location
        required_sentinel2_bands = [
            "SCL",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
        # Get the Sentinel-2 data
        s2_data = {}
        for band in required_sentinel2_bands:
            # The scene classification map is categorical, i.e. we need NN resampling
            if band == "SCL":
                resampling_method = Resampling.nearest
            else:
                resampling_method = Resampling.bilinear
            href = sentinel_2_href[band]
            with rasterio.open(href, "r") as src:
                # Compute transform
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, *src.bounds, resolution=10
                )
                with WarpedVRT(
                    src,
                    crs=src.crs,
                    transform=transform,
                    height=height,
                    width=width,
                    resampling=resampling_method,
                ) as vrt:
                    height_s2 = vrt.height
                    width_s2 = vrt.width
                    crs_s2 = vrt.crs
                    s2_window = get_window_for_footprint(
                        location, edge_length=320, transform=vrt.transform, crs=vrt.crs
                    )
                    d = vrt.read(1, window=s2_window)
                    s2_data[band] = d
                    s2_profile = vrt.profile

                    if band == "SCL":
                        # Check the SCL classes at the footprint location
                        stats, is_valid = compute_scl_stats(d)
                        if not is_valid:
                            raise ValueError(
                                "SCL classes at footprint location are not allowed"
                            )
                        # Convert SCL dict to numpy array
                        stats = np.array(list(stats.items()))

        # Get the CopDEM data
        copdem_data = {}
        with rasterio.open(copdem_hrefs["data"], "r") as src:
            transform, width, height = calculate_default_transform(
                src.crs, crs_s2, src.width, src.height, *src.bounds, resolution=10
            )
            with WarpedVRT(
                src,
                crs=crs_s2,
                transform=transform,
                height=height,
                width=width,
                resampling=Resampling.bilinear,
                nodata=src.nodata,
            ) as vrt:
                copdem_window = get_window_for_footprint(
                    location, edge_length=320, transform=vrt.transform, crs=vrt.crs
                )
                copdem_data["Elevation"] = vrt.read(1, window=copdem_window)
                # copdem_data["Slope"] , copdem_data["Aspect"]  = compute_slope_and_aspect(copdem_data["Elevation"], nodata=src.nodata, zscale=1)
                copdem_profile = vrt.profile
                copdem_data["Slope"], is_valid = compute_slope_degrees(
                    copdem_data["Elevation"], scale_z=0.1
                )
                if not is_valid:
                    raise ValueError(
                        "Slope at footprint location is above 15 perecent!"
                    )

        # Get WorldCover data
        worldcover_data = {}
        with rasterio.open(worldcover_href["map"], "r") as src:
            transform, width, height = calculate_default_transform(
                src.crs, crs_s2, src.width, src.height, *src.bounds, resolution=10
            )
            with WarpedVRT(
                src,
                crs=crs_s2,
                transform=transform,
                height=height,
                width=width,
                resampling=Resampling.nearest,
                nodata=src.nodata,
            ) as vrt:
                worldcover_window = get_window_for_footprint(
                    location, edge_length=320, transform=vrt.transform, crs=vrt.crs
                )
                worldcover_data["classe"] = vrt.read(1, window=worldcover_window)
                worldcover_profile = vrt.profile

        # Create mask from GEDI footprint
        warped_footprint = shape(
            warp.transform_geom("epsg:4326", crs_s2, [location])[0]
        )
        warped_footprint = warped_footprint.buffer(
            12
        )  # a GEDI footprint has an approx. diameter of 25 meter

        window_transform = rasterio.windows.transform(
            s2_window, s2_profile["transform"]
        )
        agbd_data = rasterize(
            [warped_footprint],
            fill=-9999,
            transform=window_transform,
            out_shape=(int(s2_window.height), int(s2_window.width)),
            all_touched=True,
            dtype=np.float32,
        )
        # Fill the AGBD value in the raster
        agbd_data[agbd_data == 1] = footprint["agbd"]

        return (
            s2_data,
            worldcover_data,
            copdem_data,
            agbd_data,
            stats,
            transform,
            crs_s2,
            s2_product_name,
            worldcover_product_name,
            copdem_product_name,
        )

    except Exception as e:
        raise e


def process_batch(
    frame: Tuple[pl.DataFrame, int],
    catalog: pystac_client.Client,
    output_dir: str,
    configurer: Callable,
    queue,
    batch_size: int,
):
    """Create a bounding box containing all footprints and find and process all matching Sentinel-2 products and metadata"""
    configurer(queue)

    df, frame_idx = frame

    logger = logging.getLogger("batch_processor")

    # Prepare arrays to store data
    batch_size = len(df)
    arr_s2 = np.empty((batch_size, 12, 32, 32), dtype=np.int16)
    arr_s2_name = np.empty((batch_size, 1), dtype="S62")

    arr_wc = np.empty((batch_size, 1, 32, 32), dtype=np.uint8)
    arr_wc_name = np.empty((batch_size, 1), dtype="S42")

    arr_cd = np.empty((batch_size, 2, 32, 32), dtype=np.float32)
    arr_cd_name = np.empty((batch_size, 1), dtype="S42")

    arr_agbd_raster = np.empty((batch_size, 1, 32, 32), dtype=np.float32)

    arr_transforms = np.empty((batch_size, 9), dtype=np.float64)
    arr_crs = np.empty((batch_size, 1), dtype="S12")

    arr_gedi_name = np.empty((batch_size, 1), dtype="S60")

    arr_agbd_value = np.empty((batch_size, 1), dtype=np.float32)
    arr_agbd_standard_error = np.empty((batch_size, 1), dtype=np.float32)
    arr_shotnumber = np.empty((batch_size, 1), dtype=np.int64)
    arr_acquisition_time = np.empty((batch_size, 1), dtype="S20")
    arr_delta_time = np.empty((batch_size, 1), dtype=np.float32)
    arr_lat_lon = np.empty((batch_size, 2), dtype=np.float64)
    arr_scl_distribution = np.empty((batch_size, 12, 2), dtype="S25")

    sucessfull_sample_count = 0
    for row in df.iter_rows(named=True):
        try:
            toi = row["acquisition_time"]
            p = Point(row["lon_lowestmode"], row["lat_lowestmode"])

            t_start = (toi - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            t_stop = (toi + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            timerange = f"{t_start}/{t_stop}"

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=p,
                datetime=timerange,
                query={"eo:cloud_cover": {"lt": 25}},
            )
            items = search.item_collection()
            item = min(items, key=lambda x: eo.ext(x).cloud_cover)

            # Read the data at footrpint location
            (
                s2_data,
                worldcover_data,
                copdem_data,
                agbd_data,
                stats,
                transform,
                crs_s2,
                s2_product_name,
                worldcover_product_name,
                copdem_product_name,
            ) = retrieve_data_for_footprint(catalog, output_dir, row, item)

            # Trasnform the data to numpy arrrays
            # Sentinel-2 data
            s2_data_np = np.empty((1, 12, 32, 32), dtype=np.int16)
            for idx, d in enumerate(s2_data.items()):
                arr = d[1]
                s2_data_np[0, idx, :, :] = arr

            arr_s2[sucessfull_sample_count, :, :, :] = s2_data_np

            # WorldCover data
            arr_wc[sucessfull_sample_count, 0, :, :] = worldcover_data["classe"]

            # CopDEM height and slope data
            copdem_data_np = np.empty((1, 2, 32, 32), dtype=np.float32)
            for idx, d in enumerate(copdem_data.items()):
                arr = d[1]
                copdem_data_np[0, idx, :, :] = arr

            arr_cd[sucessfull_sample_count, :, :, :] = copdem_data_np

            # AGBD mask
            arr_agbd_raster[sucessfull_sample_count, 0, :, :] = agbd_data

            # Non-raster data
            arr_s2_name[sucessfull_sample_count, :] = s2_product_name
            arr_wc_name[sucessfull_sample_count, :] = worldcover_product_name
            arr_cd_name[sucessfull_sample_count, :] = copdem_product_name

            arr_agbd_value[sucessfull_sample_count, :] = row["agbd"]
            arr_transforms[sucessfull_sample_count, :] = transform
            arr_crs[sucessfull_sample_count, :] = str(crs_s2)

            arr_gedi_name[sucessfull_sample_count, :] = row["GEDI Filename"]

            arr_agbd_standard_error[sucessfull_sample_count, :] = row["agbd_se"]
            arr_shotnumber[sucessfull_sample_count, :] = row["shot_number"]
            arr_acquisition_time[
                sucessfull_sample_count, :
            ] = datetime.datetime.strftime(row["acquisition_time"], "%Y-%m-%d-%H-%M-%S")
            arr_delta_time[sucessfull_sample_count, :] = row["delta_time"]
            arr_lat_lon[sucessfull_sample_count, :] = np.array(
                [row["lat_lowestmode"], row["lon_lowestmode"]]
            )
            arr_scl_distribution[sucessfull_sample_count, :, :] = stats

            sucessfull_sample_count += 1

        except Exception as e:
            logger.error(e)
            continue

    logger.info(
        f"Sucessfully created {sucessfull_sample_count} samples. Writing HDF5 subset to disc"
    )

    # Clip array to sucessfull samples and write to HDF5
    data = {
        "Sentinel-2 L2A": arr_s2[:sucessfull_sample_count, :, :, :],
        "WorldCover": arr_wc[:sucessfull_sample_count, :, :, :],
        "CopDEM": arr_cd[:sucessfull_sample_count, :, :, :],
        "AGBD Raster": arr_agbd_raster[:sucessfull_sample_count, :, :],
        "Transforms": arr_transforms[:sucessfull_sample_count, :],
        "CRS": arr_crs[:sucessfull_sample_count, :],
        "S2 SCL Stats": arr_scl_distribution[:sucessfull_sample_count, :, :],
        "Sentinel-2 Product Name": arr_s2_name[:sucessfull_sample_count, :],
        "WorldCover Tile Name": arr_wc_name[:sucessfull_sample_count, :],
        "CopDEM Tile Name": arr_cd_name[:sucessfull_sample_count, :],
        "AGBD Values": arr_agbd_value[:sucessfull_sample_count, :],
        "AGBD Standard Error": arr_agbd_standard_error[:sucessfull_sample_count, :],
        "GEDI Product Name": arr_gedi_name[:sucessfull_sample_count, :],
        "GEDI Shot Number": arr_shotnumber[:sucessfull_sample_count, :],
        "GEDI Acquisition Time": arr_acquisition_time[:sucessfull_sample_count, :],
        "GEDI Delta Time": arr_delta_time[:sucessfull_sample_count, :],
        "GEDI Lat Lon": arr_lat_lon[:sucessfull_sample_count, :],
    }
    # Generate a random filename
    filename = os.path.join(
        output_dir,
        f"batch_{batch_size}_{frame_idx}" + ".h5",
    )

    path_hdf, num_samples = numpy_to_hdf5(data, filename)

    logger.info(
        f"Sucessfully wrote HDF5 file to {path_hdf}. Number of samples: {num_samples}"
    )


def numpy_to_hdf5(arrays: dict, filename: str) -> Tuple[str, int]:
    """Writes a dictionary of numpy arrays to a HDF5 file

    The key is used as the name of the dataset. The data type will be infered from the arrays.

    Args:
        arrays: Dictionary of numpy arrays
        filename: Path to the HDF5 file
    Returns:
        Path to the HDF5 file and number of samples in file
    """
    with h5py.File(filename, "w") as f:
        grp_data = f.create_group("data")
        for key, value in arrays.items():
            chunk_shape = list(value.shape)
            chunk_shape[0] = 1  # Use chunk size of 1 for the first dimension

            datatype = value.dtype
            if datatype.kind in {"U", "S"}:
                datatype = h5py.string_dtype()  #  Use h5py string dtype for strings

            grp_data.create_dataset(
                key,
                data=value,
                compression="lzf",
                chunks=tuple(chunk_shape),
                dtype=datatype,
            )
    return filename, value.shape[0]


def create_subset(
    path_csv: str,
    output_dir: str,
    filename_plot: str,
    filename_csv: str,
    dataset_size_max: int,
) -> str:
    """Creates a subset with (attempted) even distribution along the time axis based on a CSV file containing GEDI footprints

    Args:
        path_csv: Filepath to the CSV file containing the GEDI footprints
        dataset_size_max: Maximum number of samples in the dataset (will create subset)

    Return:
        filename of the written CSV file containg the subset
    """
    df = pl.read_csv(path_csv, separator=";", try_parse_dates=True)
    # The data is not evenly distributed across the day of the year. Hence, here we'll create a more evenly distributed subset

    max_samples_per_day = int(
        (dataset_size_max * 1.5) / 365
    )  # Several days don't have many or no samples
    df_samples = pl.DataFrame()
    logging.info(f"Attempting to create evenly temporal distribution")
    for day in range(366):
        logging.info(f"Finding samples for day {day+1}/365")
        df_day = df.filter(df["acquisition_time"].dt.ordinal_day() == day)
        if len(df_day) == 0:
            continue
        if len(df_day) < max_samples_per_day:
            # We can use all samples from this day
            df_samples = pl.concat([df_samples, df_day])
        else:
            # We have too many samples from this day. We need to subset
            df_subset = df_day.sample(n=max_samples_per_day, shuffle=True, seed=0)
            df_samples = pl.concat([df_samples, df_subset])

    df_samples = df_samples.sample(frac=1.0, shuffle=True, seed=0)
    logging.info("Plotting data distribution")
    plot_gedi_footprint_distribution(df_samples, output_dir, filename_plot)

    filename_csv = os.path.join(output_dir, filename_csv)
    logging.info(f"Writing subset to disc: {filename_csv}")

    df_samples.write_csv(filename_csv, separator=";")

    return filename_csv


def plot_gedi_footprint_distribution(
    data: Union[str, pl.DataFrame], output_dir: str, filename: str
) -> None:
    """Plots the global spatial and temporal distribution of GEDI footprints.

    Args:
        data: Either path to CSV file or Polars DataFrame
        output_dir: Output directory where the plot will be saved.
        filename: Filename of the plot."""

    if type(data) == str and data.endswith(".csv"):
        df = pl.read_csv(data, separator=";", try_parse_dates=True)
    elif type(data) == pl.DataFrame:
        df = data
    dfn = df.select(["lat_lowestmode", "lon_lowestmode"]).to_numpy()

    f, axs = plt.subplots(2, 1, dpi=200)

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.boundary.plot(ax=axs[0], facecolor="none", edgecolor="black")

    y = dfn[:, 0]  # latitude
    x = dfn[:, 1]  # longitude

    axs[0].plot(x, y, "or", markersize=0.1, alpha=0.05)
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    axs[0].set_title("Spatial distribution")

    # Temporal distribution
    df = df.with_columns(
        pl.col("acquisition_time").dt.ordinal_day().alias("Day of Year")
    )

    dfn = df.groupby("Day of Year").count().sort("Day of Year").to_numpy()

    x = dfn[:, 0]  # Day of year
    y = dfn[:, 1]  # Footprint count

    axs[1].bar(x=x, height=y)
    axs[1].set_xlabel("Day of Year")
    axs[1].set_ylabel("Footprint Count")
    axs[1].set_title("Temporal distribution")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, filename)
    f.savefig(filename, dpi="figure")


def create_combined_csv(
    input_file_list: str, output_dir: str, output_filename: str
) -> Tuple[str, str]:
    """Creates a combined CSV file from all HDF5 files in the input directory.

    Args:
        input_file_list: List of paths to the input HDF5 files.
        output_dir: Path to the output directory where the CSV file will be saved.
        output_filename: Name of the output CSV file.

    Returns:
        Path to the output CSV file."""
    # Create an aggregated CSV file containing the combined data from all provided GEDI HFD5 files
    df_train = pl.DataFrame()
    df_test = pl.DataFrame()
    for idx, file in enumerate(input_file_list):
        filename = os.path.basename(file)
        logging.info(f"Processing HDF5 file {idx}/{len(input_file_list)}: {filename}")
        df_current = hdf5_to_dataframe(file)
        logging.info(
            f"Finished processing HDF5 file: {filename}. Retrieved {len(df_current)} rows."
        )

        year = int(re.findall(r"\d{4}", filename)[0])
        if year not in (2020, 2021):
            raise ValueError(f"Invalid year {year} in filename {filename}.")

        if year == 2020:
            df_train = pl.concat([df_train, df_current])
        elif year == 2021:
            df_test = pl.concat([df_test, df_current])

    # Shuffle the contents of the dataframe to mix footprints from different files
    df_train = df_train.sample(fraction=1.0, shuffle=True, seed=0)
    df_test = df_test.sample(fraction=1.0, shuffle=True, seed=0)

    # Write the dataframe to a CSV file
    logging.info(f"Writing combined CSV files to {output_dir}.")
    filepath_train = os.path.join(output_dir, "train_" + output_filename)
    filepath_test = os.path.join(output_dir, "test_" + output_filename)
    df_train.write_csv(filepath_train, separator=";")
    df_test.write_csv(filepath_test, separator=";")

    return filepath_train, filepath_test


def hdf5_to_dataframe(hdf5_path: str) -> pl.DataFrame:
    """This function takes a path to an HDF5 file and returns a polars DataFrame with the contents of the file."""
    with h5py.File(hdf5_path, "r") as f:
        beams = [key for key in f.keys() if key[:4] == "BEAM"]
        fields_to_use = [  # https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html
            "agbd",
            "agbd_se",
            "lon_lowestmode",
            "lat_lowestmode",
            "shot_number",
            "delta_time",
            "l2_quality_flag",  # Flag identifying the most useful L2 data for biomass predictions
            "l4_quality_flag",  # Flag simplifying selection of most useful biomass predictions
            "surface_flag",  # Indicates elev_lowestmode is within 300m of Digital Elevation Model (DEM) or Mean Sea Surface (MSS) elevation
        ]

        data = pl.DataFrame()

        for beam in beams:
            beam_data = {}
            for key in fields_to_use:
                beam_data[key] = f[beam][key][:]

            data = pl.concat([data, pl.DataFrame(beam_data)])

    # Filter data using the provided quality flags
    data = data.filter(
        (pl.col("l2_quality_flag") == 1)
        & (pl.col("l4_quality_flag") == 1)
        & (pl.col("surface_flag") == 1)
    )
    # Filter data using the standard error (based on "Fusing GEDI with earth observation data for large area aboveground biomass mapping")
    data = data.filter((pl.col("agbd_se") / pl.col("agbd")) * 100 < 50)

    # Add GEDI source file name for later reference and get acquistion date and time from filename
    src_filename = os.path.basename(hdf5_path)
    julien_datetime_string = src_filename.split("_")[2]
    acquisition_time = datetime.datetime.strptime(julien_datetime_string, "%Y%j%H%M%S")
    data = data.with_columns(
        pl.lit(src_filename).alias("GEDI Filename"),
        pl.lit(acquisition_time).alias("acquisition_time"),
    )

    return data


def save_gedi_mask(
    output_dir: str,
    agbd_value: float,
    shot_number: int,
    year: int,
    profile: rasterio.profiles.Profile,
    window: windows.Window,
    location: Point,
    crs: int,
):
    """Save the rasterized GEDI footprint.

    The raster has the same dimensions as the data windos and contains the AGBD values at the pixel locations that intersect with the GEDI footint.
    All other non-intersecting pixels have no data values of -9999.

    Args:
        output_dir: Full path to directory where to store the file
        agbd_value: AGBD value of the GEDI footprint
        shot_number: Number of the GEDI shot the window refers to. Will be used for filenaming
        year: Year of the footprint
        profile: rasterio.profiles.Profile of the warped source dataset
        window: The window to use for writing the data
        footprint: Location of the GEDI footprint as Point
        crs: Target crs identifier to use

    Returns:
        filepath: Filepath of the written file
    """
    # First we need to reproject the footprint to UTM
    warped_footprint = shape(warp.transform_geom("epsg:4326", crs, [location])[0])
    warped_footprint = warped_footprint.buffer(
        12
    )  # a GEDI footprint has an approx. diameter of 25 meter

    window_transform = rasterio.windows.transform(window, profile["transform"])
    raster = rasterize(
        [warped_footprint],
        fill=-9999,
        transform=window_transform,
        out_shape=(int(window.height), int(window.width)),
        all_touched=True,
        dtype=np.float32,
    )
    # Fill the AGBD value in the raster
    raster[raster == 1] = agbd_value
    meta = profile.copy()
    meta["width"] = int(window.width)
    meta["height"] = int(window.height)
    meta["transform"] = window_transform
    meta["count"] = 1
    meta["driver"] = "COG"
    meta["dtype"] = "float32"

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{year}_{shot_number}_agbd_mask.tif")

    with rasterio.open(filename, "w", **meta) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, "AGBD")

    return filename


def save_window_data(
    output_dir: str,
    shot_number: int,
    product_name: str,
    year: int,
    profile: rasterio.profiles.Profile,
    window: windows.Window,
    data: dict,
    dtype: str,
) -> str:
    """Save the data from a window to a mulitband GeoTIFF file. The filename will be <year>_<GEDI shot number>_<product name>.tif

    Args:
        output_dir: Full path to directory where to store the file
        shot_number: Number of the GEDI shot the window refers to. Will be used for filenaming in addition to the product name
        product_name: Name of the product the window is from. Used for file naming
        year: Year of the footprint
        profile: rasterio.profiles.Profile of the warped source dataset
        window: The window to use for writing the data
        data: Dict containing the raster data with the band names as keys
        dtype: Raster data type to use

    Returns:
        filepath: Filepath of the written file
    """
    window_transform = rasterio.windows.transform(window, profile["transform"])

    meta = profile.copy()
    meta["width"] = int(window.width)
    meta["height"] = int(window.height)
    meta["transform"] = window_transform
    meta["count"] = len(data.keys())
    meta["driver"] = "COG"
    meta["dtype"] = dtype

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{year}_{shot_number}_{product_name}.tif")

    with rasterio.open(filename, "w", **meta) as dst:
        for band_id, band in enumerate(data.keys()):
            dst.write(data[band], band_id + 1)
            dst.set_band_description(band_id + 1, band)

    return filename


def get_window_for_footprint(
    location: Point,
    edge_length: int,
    transform: rasterio.DatasetReader.transform,
    crs: rasterio.CRS,
) -> windows.Window:
    """Generate a rectangular window with a specific width and heigth centered at a given location.

    Args:
        location: Shaply Point defining the location in EPSG:4326
        edge_length: Edge length of the window in meters
        trasnform: Transform of the target raster
        crs: Rasterio CRS to be used

    Returns:

    """
    aoi_bounds = features.bounds(location)

    # Reproject the geometry
    warped_aoi_bounds = warp.transform_bounds("epsg:4326", crs, *aoi_bounds)
    # Add some offset in meters to have a rectangle and not a point
    offset = round(edge_length / 2, 0)  # m
    warped_aoi_bounds = (
        warped_aoi_bounds[0] - offset,
        warped_aoi_bounds[1] - offset,
        warped_aoi_bounds[2] + offset,
        warped_aoi_bounds[3] + offset,
    )
    window = windows.from_bounds(transform=transform, *warped_aoi_bounds)

    return window


def get_sentinel2_hrefs(catalog: pystac_client.Client, item_id: str) -> dict:
    """Reads all available assets for a PyStac Client and returns their names and hrefs as a Python dictionary.

    Args:
        catalog: PySTAC catalog
        item_id: item_id of the item to unpack
    """
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = search.item_collection()
    if len(items) != 1:
        raise ValueError(f"Item {item_id} returns more than one item.")
    hrefs = {key: asset.href for key, asset in items[0].assets.items()}
    return hrefs


def get_aux_hrefs_for_location(
    catalog: pystac_client.Client, location: Point, year: int
) -> Tuple[dict, dict]:
    """Finds the intersecting WorldCover and CopDEM tiles and returns their asset hrefs based on a point in EPSG:4326.

    Args:
        catalog: PySTAC catalog
        location: Shaply Point defining the location in EPSG:4326
        year: Year to use. Relevant for retrieving WorldCover data

    Returns:
        Tuple of (copdem_refs, worldcover_hrefs)
    """
    # Retrieve WorldCover item
    search = catalog.search(
        collections=["esa-worldcover"],
        intersects=location,
    )
    wc_items = search.item_collection()
    # Filter year
    year_to_use = 2020 if year <= 2020 else 2021
    wc_items = [item for item in wc_items if item.id.split("_")[3] == str(year_to_use)]
    if len(wc_items) > 1:
        logging.warning(
            f"Found more than one WorldCover tile for location. Skipping location."
        )
        return False, False

    wc_hrefs = {key: asset.href for key, asset in wc_items[0].assets.items()}

    # CopDEM
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        intersects=location,
    )
    dem_items = search.item_collection()
    if len(dem_items) > 1:
        logging.warning(
            f"Found more than one CopDEM tile for location. Skipping location."
        )
        return False, False
    dem_hrefs = {key: asset.href for key, asset in dem_items[0].assets.items()}

    return dem_hrefs, wc_hrefs


def read_window_around_point(
    href: str,
    location: Point,
    window_width: int,
    window_height: int,
    pixel_size_meter: int,
    resample_algorithm: str,
) -> np.ndarray:
    """Opens a single band GeoTIFF file and creates a window with the desired parameters that is centered around the specified longitude and latitude.

    Args:
        href: href of the file to read
        Point: Shapely Point defining the latitude and longitude of the window center in EPSG:4326
        window_width: Width of the returned window in pixels
        window_height: Height of the provided windows in pixels
        pixel_size_meter: Size of the a single pixel in meters
        resample_algorithm: Resample algorithm to use. Must be one of rasterio.enums.Resampling

    Returns:
        A Numpy ndarray containing the window data
    """
    with rasterio.open(href) as ds:
        aoi_bounds = features.bound(location)
        # Reproject the geometry
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        # Add some offset in meters to have a rectangle and not a point
        offset = 80  # m
        warped_aoi_bounds = (
            warped_aoi_bounds[0] - offset,
            warped_aoi_bounds[1] - offset,
            warped_aoi_bounds[2] + offset,
            warped_aoi_bounds[3] + offset,
        )
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        band_data = ds.read(window=aoi_window)
