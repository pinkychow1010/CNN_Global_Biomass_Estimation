# Automated Dataset Generation

The scripts in this module can be used to create a ML ready dataset based on a set of GEDI input files. 

**Start the process with*:*
```
python create_dataset.py --input <directory containing GEDI HDF5 files> --output <output directory>
```


The general workflow is the following (all output is written to the `--output` directory):
1. One or more GEDI HDF5 files must be provided in the `--input` directory
2. The GEDI data from all files is combined in a single dataframe, shuffled and stored as `train_combined.csv` (data from 2020) or `test_combined.csv` (data from 2021)
    - Only footprints that have the `l2_quality_flag`, `l4_quality_flag` and `surface_flag` set and that have a standard agbd error that is less than half the agbd estimate are kept
    data = data.filter((pl.col("agbd_se") / pl.col("agbd")) * 100 > 50)
    - If these files already exist you can skip the first two steps by providing the `--path-aggregated-csv-train` and `--path-aggregated-csv-test` arguments
    - You can create a plot showing the spatial distribution of the data by enabling the `--plot-distribution` flag
3. A subset of footprints is selected (size is controlled by the `dataset_size_max` parameter for `csv_subset_train()`). Here an even temporal distribution of the samples is attempted (only works partially)
    - Two plots are generated that show the spatial and the temporal distribution of the data: `train_distribution_subset.png` and `test_distribution_subset.png`
    - This step can also be skipped by providing the `--path-subset-csv-train` and `--path-subset-csv-test` arguments, if the files already exist
4. In the next step the created subset is split into patches of a given size (e.g. 1000 footprints). These patches are then processed by multiple processes (controlled by the `num_processors` and `batch_size` arguments of  `create_hdf_dataset()`). For each footprint:
    - The least cloudy Sentinel-2 L2A product within +-10 days is found (current cloudiness threshold is 25 %)
    - The level 2 scene classification (SLC) is check to make sure the footprint is located on pixels that are either `vegetated` or `non-vegetated`. Several other checks are done `compute_scl_stats()`. If the footprint is valid, Sentinel-2 data in a 32x32 window (10 m GSD) around the footprint is extarcted, along with and the window statistics of the SCL band.
    - Based on the window the corresponding CopDEM data is found. The slope is computed and added. If the slope at the footprint location is higher than 15 % the footprint is skipped (GEDI doesn't like sloped terrain)
    - The ESA WorldCover daata for the window is retrived
    - All data is combined in a HDF5 file (one for each batch)
5. A final step combines all created HDF5 files into a single large file. Run `create_dataset.py <output directory> --fuse-hdf5-datasets <directory containg the HDF5 files to fuse>`. 


The structure of the files from the batch and the final combined file is identical and looks like this:
```
--<train/test>_dataset.h5
    | data
    |- Sentinel-2 L2A               -> (uint16)  Raster data containing the Sentinel-2 bands. Shape: (length dataset, 12, 32, 32). Band order: SCL,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
    |- WorldCover                   -> (uint8)   ESA WorldCover data                          Shape: (length dataset, 1, 32, 32)
    |- CopDEM                       -> (float32) ESA CopDEM data (altitude [m] and slope [%]) Shape: (length dataset, 2, 32, 32)
    |- AGBD Raster                  -> (float32) GEDI AGBD value rasterized                   Shape: (length dataset, 1, 32, 32)
    |- Transforms                   -> (float32) Affine transform of the window               Shape: (length dataset, 9)
    |- CRS                          -> (string)  EPSG CRS identifer of the window             Shape: (length dataset, 1)
    |- S2 SCL Stats                 -> (string)  SCL class distribution                       Shape: (length dataset, 12, 2)
    |- Sentinel-2 Product Name      -> (string)  Sentinel-2 product name                      Shape: (length dataset, 1)
    |- WorldCover Tile Name         -> (string)  WorldCover tile name                         Shape: (length dataset, 1)
    |- CopDEM Tile Name             -> (string)  CopDEM tile name                             Shape: (length dataset, 1)
    |- AGBD Values                  -> (float32) AGBD value of the GEDI footprint             Shape: (length dataset, 1)
    |- AGBD Standard Error          -> (float32) AGBD standard error of the GEDI footprint    Shape: (length dataset, 1)
    |- GEDI Product Name            -> (string)  GEDI source product name                     Shape: (length dataset, 1)
    |- GEDI Shot Number             -> (int64)   GEDI footprint shotnumber                    Shape: (length dataset, 1)
    |- GEDI Acquisition Time        -> (string)  Date and time of GEDI acquisition (coarse)   Shape: (length dataset, 1)
    |- GEDI Delta Time              -> (string)  Time since 2018-01-01 00:00 (exact)          Shape: (length dataset, 1)
    |- GEDI Lat Lon                 -> (float32) Latitude and longitude of the foootprint      Shape: (length dataset, 2)
```