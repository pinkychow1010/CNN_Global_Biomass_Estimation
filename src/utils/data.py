# imports
import os
import pandas as pd
import geopandas as gpd
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer
import rasterio
from rasterio import windows, features, warp
import datetime
import json

import shapely
from shapely.geometry import Point

# import rich.table
from typing import Tuple


def get_time_info_gedi(delta_time):
    # timestamp for the chosen GEDI shot
    base_date = datetime.datetime(2018, 1, 1, 0, 0, 0)
    timestamp = base_date + datetime.timedelta(seconds=delta_time)
    time_of_interest = timestamp.date()

    return time_of_interest


def open_catalog():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog


def find_matching_sentinel_products(catalog, lats, lons, tois, cloud_threshold):
    """Create a bounding box containing all footprints and fina all matching Sentinel-2 products"""
    # Planetary computer data access
    # Search all intersecting files

    """bbox = [min(lons), min(lats), max(lons), max(lats)]
    
    intersecting_items = []
    points = []
    
    for idx in range(len(lats)):
        timerange = f"{tois[idx] - timedelta(days=10)}/{tois[idx] + timedelta(days=10)}"
        point = Point(lons[idx], lats[idx])
        points.append(point)
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=point,
            datetime=timerange,
            query={"eo:cloud_cover": {"lt": cloud_threshold}}
            )

        items = search.item_collection()
        if(len(items) > 0):
            intersecting_items.append(min(items, key=lambda item: eo.ext(item).cloud_cover))
        else:
            intersecting_items.append(None)

    return  intersecting_items, points"""

    timerange = f"{tois[0] - timedelta(days=10)}/{tois[0] + timedelta(days=10)}"
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=timerange,
        query={"eo:cloud_cover": {"lt": cloud_threshold}},
    )

    items = search.item_collection()

    return items


def find_intersecting_products(lat, lon, items):
    """Find the intersecting products for a single point given a list of PySTAC items"""
    polygons = [shapely.geometry.box(*item.bbox, ccw=True) for item in items]
    point = Point(lon, lat)
    intersecting_items = []
    for idx, poly in enumerate(polygons):
        # if poly.contains(point):
        if poly.intersects(point):
            intersecting_items.append(items[idx])

    return intersecting_items, point


def check_pixel_is_cloud(item, feature, shot_number):
    """Reads scene classification mask at a given location and checks if the pixels is marked as cloudy.

    Args:
        item: Pystac Item Sentinel-2 object to read
    """

    href = item.assets["SCL"].href
    with rasterio.open(href) as ds:
        aoi_bounds = features.bounds(feature)
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

        save_window_data(shot_number, ds, aoi_window, band_data)

        cloudy = 0

        wanted_bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B09",
            "B11",
            "B12",
            "B8A",
            "SCL",
        ]
        hrefs = {}
        for band in wanted_bands:
            hrefs[band] = item.assets[band].href

        if band_data.size != 64:
            is_valid_pixel = False
        else:
            center_pixels = band_data[:1, 3:5, 3:5]
            r = (center_pixels >= 4) & (center_pixels <= 6)
            valid_mid_pixel = np.all(r)

            # band = band_data.copy()
            # band[:1,3:5,3:5] = 100

            # print(band_data)
            # print(center_pixels)
            # print('valid_mid_pixel: ', valid_mid_pixel)

            if valid_mid_pixel:
                for x in np.nditer(band_data):
                    if (x == 2) | (x == 3) | (x == 8) | (x == 9) | (x == 10):
                        cloudy += 1
                    elif (x == 4) | (x == 5) | (x == 6):
                        continue
                    elif (x == 0) | (x == 1) | (x == 7) | (x == 11):
                        is_valid_pixel = False
                        return is_valid_pixel, band_data, cloudy, hrefs
            else:
                # print('pixels around footprint are not valid')
                is_valid_pixel = False
                return is_valid_pixel, band_data, cloudy, hrefs

            if cloudy <= 0.15 * band_data.size:
                # print('cloudy: ', cloudy)
                valid_cloudy = True
            else:
                # print('cloudy: * ', cloudy)
                valid_cloudy = False

            is_valid_pixel = valid_mid_pixel & valid_cloudy

            # print("total pixels: ", band_data.size)
            # print("final result: ", is_valid_pixel)

        return is_valid_pixel, band_data, cloudy, hrefs


def check_pixels(item, feature, shot_number):
    # print(f'Pixels details for shot number {shot_number}: ')

    is_valid_pixel, pixel_values, cloud_pixels, hrefs = check_pixel_is_cloud(
        item, feature, shot_number
    )

    # print('were the pixel values valid? ', is_valid_pixel)

    if is_valid_pixel:
        vegetation = np.count_nonzero(pixel_values == 4) / pixel_values.size
        non_vegetated = np.count_nonzero(pixel_values == 5) / pixel_values.size
        water = np.count_nonzero(pixel_values == 6) / pixel_values.size
        cloud = cloud_pixels / pixel_values.size

        # print(vegetation, non_vegetated, water, cloud)

        return vegetation, non_vegetated, water, cloud, hrefs
    else:
        # print('footprint should not be included')
        return -1, -1, -1, -1, {}


def save_window_data(shot_number, ds, window, data):
    """Save the data from a window to a GeoTIFF file"""

    window_transform = rasterio.windows.transform(window, ds.profile["transform"])

    meta = ds.profile.copy()
    meta["width"] = int(window.width)
    meta["height"] = int(window.height)
    meta["transform"] = window_transform

    filename = f"{shot_number}_window.tif"

    with rasterio.open(filename, "w", **meta) as dst:
        dst.write(data)


def write_geojson(filename, props):
    """Writes the output of chunk to a GeoJSON file"""
    features = []
    props = list(props)
    for feature in props:
        (
            shot,
            agbd,
            lon,
            lat,
            toi,
            l2_flag,
            l4_flag,
            vegetation_pixels,
            non_vegetated_pixels,
            water_pixels,
            cloud_pixels,
            hrefs,
        ) = feature

        f = {
            "type": "feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {
                "shotnumber": str(shot),
                "agbd": float(agbd),
                "time": str(toi),
                "l2_quality_flag": str(l2_flag),
                "l4_quality_flag": str(l4_flag),
                "vegetation_pixels": str(vegetation_pixels),
                "non_vegetated_pixels": str(non_vegetated_pixels),
                "water_pixels": str(water_pixels),
                "cloud_pixels": str(cloud_pixels),
                "B02": str(hrefs["B02"]),
                "B03": str(hrefs["B03"]),
                "B04": str(hrefs["B04"]),
                "B05": str(hrefs["B05"]),
                "B06": str(hrefs["B06"]),
                "B07": str(hrefs["B07"]),
                "B08": str(hrefs["B08"]),
                "B8A": str(hrefs["B8A"]),
                "B09": str(hrefs["B09"]),
                "B11": str(hrefs["B11"]),
                "B12": str(hrefs["B12"]),
                "SCL": str(hrefs["SCL"]),
            },
        }
        features.append(f)

    content = {"type": "FeatureCollection", "Features": features}
    with open(filename, "w") as f:
        json.dump(content, f)
