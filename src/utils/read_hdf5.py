import h5py
from multiprocessing import Process
import multiprocessing
from azure.storage.blob import BlobServiceClient
from utils.data import *
from typing import List
import numpy as np
import datetime
import logging as log

log.basicConfig(level=log.WARNING)


def get_data(chunk: List, chunk_number, beam, filename, dir_out):
    """Process a chunk of GEDI footprints. Will be executed by a new thread using multiprocessing.

    Args:
        chunk: List of various parameters from the GEDI file (with the length of the specified chunk size)
        chunk_number: Number of current chunk. Used for filenaming
        beam: Beam of the current chunk
        dir_out: Output directory were the results will be written too

    """

    agbds = chunk[0]
    lons = chunk[1]
    lats = chunk[2]
    shots = chunk[3]
    ts = chunk[4]
    tois = chunk[5]
    l2flag = chunk[6]
    l4flag = chunk[7]

    """agbds  = agbds[:100]
    lons = lons[:100]
    lats = lats[:100]
    shots = shots[:100] 
    ts = ts[:100]
    tois = tois[:100] 
    l2flag = l2flag[:100]
    l4flag = l4flag[:100]"""

    products = []
    points = []

    # Open STAC catalog
    catalog = open_catalog()

    # Find matching products for current chunk
    # items, points = find_matching_sentinel_products(catalog=catalog, lats=lats, lons=lons, tois=tois, cloud_threshold=50)
    items = find_matching_sentinel_products(
        catalog=catalog, lats=lats, lons=lons, tois=tois, cloud_threshold=50
    )
    counter_no_intersecting_product = 0

    """products = items
    remove_idx = []
    
    for idx, item in enumerate(items):
        if item is None:
            remove_idx.append(idx)"""

    if len(items) > 0:
        remove_idx = []

        for footprint in range(len(lons)):
            intersecting_items, point = find_intersecting_products(
                lats[footprint], lons[footprint], items
            )

            if len(intersecting_items) > 0:
                least_cloudy_item = min(
                    intersecting_items, key=lambda item: eo.ext(item).cloud_cover
                )
                products.append(least_cloudy_item)
                points.append(point)

            else:
                remove_idx.append(footprint)
                counter_no_intersecting_product += 1
        print(
            f"File: {filename}\tChunk Number: {chunk_number}\tNumber of found products: {len(items)}\t Intersecting: {len(products)} \t No inter: {counter_no_intersecting_product}"
        )

        if len(remove_idx) > 0:
            agbds = np.delete(agbds, remove_idx)
            lons = np.delete(lons, remove_idx)
            lats = np.delete(lats, remove_idx)
            shots = np.delete(shots, remove_idx)
            ts = np.delete(ts, remove_idx)
            tois = np.delete(tois, remove_idx)
            l2flag = np.delete(l2flag, remove_idx)
            l4flag = np.delete(l4flag, remove_idx)
            products = np.delete(products, remove_idx)
            points = np.delete(points, remove_idx)

        assert (
            len(agbds)
            == len(lons)
            == len(lats)
            == len(shots)
            == len(ts)
            == len(tois)
            == len(l2flag)
            == len(l4flag)
            == len(products)
            == len(points)
        )

    else:
        print(
            f"No matching item found for beam {beam}, chunk {chunk_number} for file {filename}"
        )

    results = None
    delete_idx = []

    if len(products) > 0:
        assert (
            len(agbds)
            == len(lons)
            == len(lats)
            == len(shots)
            == len(ts)
            == len(tois)
            == len(l2flag)
            == len(l4flag)
            == len(products)
            == len(points)
        )

        results = list(map(check_pixels, products, points, shots))

        fraction_of_vegetation_pixels = []
        fraction_of_non_vegetated_pixels = []
        fraction_of_water_pixels = []
        fraction_of_cloud_pixels = []
        hrefs = []

        for idx, ele in enumerate(results):
            if (ele[0] == -1) | (ele[1] == -1) | (ele[2] == -1) | (ele[3] == -1):
                delete_idx.append(idx)

            fraction_of_vegetation_pixels.append(ele[0])
            fraction_of_non_vegetated_pixels.append(ele[1])
            fraction_of_water_pixels.append(ele[2])
            fraction_of_cloud_pixels.append(ele[3])
            hrefs.append(ele[4])

        if len(delete_idx) > 0:
            agbds = np.delete(agbds, delete_idx)
            lons = np.delete(lons, delete_idx)
            lats = np.delete(lats, delete_idx)
            shots = np.delete(shots, delete_idx)
            ts = np.delete(ts, delete_idx)
            tois = np.delete(tois, delete_idx)
            l2flag = np.delete(l2flag, delete_idx)
            l4flag = np.delete(l4flag, delete_idx)
            products = np.delete(products, delete_idx)
            points = np.delete(points, delete_idx)
            fraction_of_vegetation_pixels = np.delete(
                fraction_of_vegetation_pixels, delete_idx
            )
            fraction_of_non_vegetated_pixels = np.delete(
                fraction_of_non_vegetated_pixels, delete_idx
            )
            fraction_of_water_pixels = np.delete(fraction_of_water_pixels, delete_idx)
            fraction_of_cloud_pixels = np.delete(fraction_of_cloud_pixels, delete_idx)
            hrefs = np.delete(hrefs, delete_idx)

        assert (
            len(agbds)
            == len(lons)
            == len(lats)
            == len(shots)
            == len(ts)
            == len(tois)
            == len(l2flag)
            == len(l4flag)
            == len(products)
            == len(points)
            == len(fraction_of_vegetation_pixels)
            == len(fraction_of_non_vegetated_pixels)
            == len(fraction_of_water_pixels)
            == len(fraction_of_cloud_pixels)
            == len(hrefs)
        )

        filename = os.path.join(
            dir_out, f"{filename}_{beam}_chunk_{chunk_number}.geojson"
        )

        write_geojson(
            filename,
            zip(
                list(shots),
                list(agbds),
                list(lons),
                list(lats),
                list(tois),
                list(l2flag),
                list(l4flag),
                fraction_of_vegetation_pixels,
                fraction_of_non_vegetated_pixels,
                fraction_of_water_pixels,
                fraction_of_cloud_pixels,
                hrefs,
            ),
        )


def data_multiprocessing(file, filename):
    DIR_OUT = "./test_chunks"
    os.makedirs(DIR_OUT, exist_ok=True)
    f = h5py.File(file, "r")

    n_cores = multiprocessing.cpu_count()
    print("Cores: ", n_cores)

    beams = [key for key in f.keys() if key[:4] == "BEAM"]
    total_footprints = 0

    for beam in beams:
        d_agbd = f[beam]["agbd"]
        d_lon = f[beam]["lon_lowestmode"]
        d_lat = f[beam]["lat_lowestmode"]
        d_geoloc = f[beam]["geolocation"]
        d_shot = f[beam]["shot_number"]
        d_time = f[beam]["delta_time"]
        d_l2_quality_flag = f[beam]["l2_quality_flag"]
        d_l4_quality_flag = f[beam]["l4_quality_flag"]

        num_items = len(d_lat)
        total_footprints += num_items
        chunk_size = 10000
        idx = 0
        # load the data in chunks of 10k rows
        for chunk_start in range(0, num_items, chunk_size * 4):
            log.info(
                f"Beam {beam}\tChunk: {chunk_start} - {chunk_start + chunk_size*4}"
            )
            chunks = []
            for chunk_start_inner in range(chunk_start, chunk_size * 4, chunk_size):
                data_chunk = [None] * 8
                chunk_stop = chunk_start_inner + chunk_size
                agbds = d_agbd[chunk_start_inner:chunk_stop][:]
                lons = d_lon[chunk_start_inner:chunk_stop][:]
                lats = d_lat[chunk_start_inner:chunk_stop][:]
                shots = d_shot[chunk_start_inner:chunk_stop][:]
                ts = d_time[chunk_start_inner:chunk_stop][:]
                tois = list(map(get_time_info_gedi, ts))[:]
                l2flag = d_l2_quality_flag[chunk_start_inner:chunk_stop][:]
                l4flag = d_l4_quality_flag[chunk_start_inner:chunk_stop][:]

                remove_idx = []

                for idx in range(len(agbds)):
                    if (
                        (agbds[idx] == -9999.0)
                        | (l2flag[idx] == 0)
                        | (l4flag[idx] == 0)
                    ):
                        remove_idx.append(idx)

                agbds = np.delete(agbds, remove_idx)
                lons = np.delete(lons, remove_idx)
                lats = np.delete(lats, remove_idx)
                shots = np.delete(shots, remove_idx)
                ts = np.delete(ts, remove_idx)
                tois = np.delete(np.array(tois), remove_idx)
                l2flag = np.delete(l2flag, remove_idx)
                l4flag = np.delete(l4flag, remove_idx)

                assert (
                    len(agbds)
                    == len(lons)
                    == len(lats)
                    == len(shots)
                    == len(ts)
                    == len(tois)
                    == len(l2flag)
                    == len(l4flag)
                )
                if len(agbds) > 0:
                    data_chunk[0] = agbds
                    data_chunk[1] = lons
                    data_chunk[2] = lats
                    data_chunk[3] = shots
                    data_chunk[4] = ts
                    data_chunk[5] = tois
                    data_chunk[6] = l2flag
                    data_chunk[7] = l4flag
                    chunks.append(data_chunk.copy())
                else:
                    chunks.append(None)

            if len(chunks) > 0:
                # Start multiple threads

                if chunks[0] is not None:
                    p1 = Process(
                        target=get_data, args=(chunks[0], 1, beam, filename, DIR_OUT)
                    )
                    p1.start()
                    # p1.join()
                if chunks[1] is not None:
                    p2 = Process(
                        target=get_data, args=(chunks[1], 2, beam, filename, DIR_OUT)
                    )
                    p2.start()
                if chunks[2] is not None:
                    p3 = Process(
                        target=get_data, args=(chunks[2], 3, beam, filename, DIR_OUT)
                    )
                    p3.start()
                if chunks[3] is not None:
                    p4 = Process(
                        target=get_data, args=(chunks[3], 4, beam, filename, DIR_OUT)
                    )
                    p4.start()


if __name__ == "__main__":
    # File must be started with this, otherwise multiprocessing won't work
    time_start = datetime.datetime.now()

    # First we need to provide the access information for our Azure container.
    # Then we can download the files and store them locally in a temporary directory.
    STORAGEACCOUNTURL = ""
    STORAGEACCOUNTKEY = ""
    CONTAINERNAME = "biomass"

    # download from blob
    blob_service_client_instance = BlobServiceClient(
        account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY
    )

    container = blob_service_client_instance.get_container_client(CONTAINERNAME)
    blob_list = container.list_blobs()

    # file_count = 0
    training_data = []
    test_data = []

    for blob in blob_list:
        if "gedi_data/" in blob.name:
            print("Processing file:\t" + blob.name)

            # file_count += 1
            # if(file_count == 1):
            # continue

            BLOBNAME = blob.name
            FILENAME = BLOBNAME
            gedi_file_name = BLOBNAME.replace("gedi_data/", "").replace(".h5", "")

            os.makedirs("./tmp/gedi_data/", exist_ok=True)
            FILEPATH = os.path.join("./tmp/", FILENAME)

            blob_client_instance = blob_service_client_instance.get_blob_client(
                CONTAINERNAME, BLOBNAME, snapshot=None
            )

            with open(FILEPATH, "wb") as my_blob:
                blob_data = blob_client_instance.download_blob()
                blob_data.readinto(my_blob)

            data_multiprocessing(FILEPATH, gedi_file_name)

            # if(file_count == 2):
            # break

        elif "processed_dataset/" in blob.name:
            # print("GeoJSON:\t" + blob.name)

            BLOBNAME = blob.name
            FILENAME = BLOBNAME
            geojson_file = BLOBNAME.replace("processed_data/", "")

            os.makedirs("./tmp/processed_dataset/", exist_ok=True)
            FILEPATH = os.path.join("./tmp/", FILENAME)

            blob_client_instance = blob_service_client_instance.get_blob_client(
                CONTAINERNAME, BLOBNAME, snapshot=None
            )

            with open(FILEPATH, "wb") as my_blob:
                blob_data = blob_client_instance.download_blob()
                blob_data.readinto(my_blob)

            if "GEDI04_A_2020" in geojson_file:
                df1 = gpd.read_file(FILEPATH)
                training_data.append(df1)
            elif "GEDI04_A_2021" in geojson_file:
                df2 = gpd.read_file(FILEPATH)
                test_data.append(df2)

    training_gdf = pd.concat(training_data)

    test_gdf = pd.concat(test_data)
    temp_cols = test_gdf.columns.tolist()
    new_cols = temp_cols[1:] + temp_cols[0:1]
    test_gdf = test_gdf[new_cols]
