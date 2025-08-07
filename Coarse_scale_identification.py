# -*- coding: utf-8 -*-
"""
Cluster lightning stroke binary 3D data using the CC3D (connected-components) algorithm.
This script reads 3D raster data, performs 26-connected component labeling,
and saves the output as a NetCDF file with appropriate spatial metadata.
Dependencies:
    - cc3d
    - numpy
    - xarray
    - rioxarray
Author: [Manxing Shi]
Date: [August 7, 2025]
"""

import cc3d
import numpy as np
import xarray as xr
import rioxarray


def get_spatial_reference(tif_path):
    """
    Retrieve X, Y coordinates and CRS from a reference GeoTIFF file.

    Parameters:
        tif_path (str): Path to the reference GeoTIFF file.

    Returns:
        tuple: (X array, Y array, CRS object)
    """
    ref = rioxarray.open_rasterio(tif_path, masked=True)
    return ref.x.values, ref.y.values, ref.rio.crs


def cluster_lightning_data(input_nc_path, reference_tif_path, output_nc_path):
    """
    Apply CC3D clustering on binary 3D lightning stroke data and save result as NetCDF.

    Parameters:
        input_nc_path (str): Path to input NetCDF file containing binary stroke data.
        reference_tif_path (str): Path to a reference GeoTIFF for CRS and coordinates.
        output_nc_path (str): Path where the clustered NetCDF will be saved.
    """
    # Read spatial reference info
    X_coords, Y_coords, crs = get_spatial_reference(reference_tif_path)

    # Load lightning binary dataset
    ds = xr.open_dataset(input_nc_path)
    stroke_binary = ds['Stroke'].data
    time_coords = ds.time
    y_coords = ds.y
    x_coords = ds.x

    # Apply 3D connected component labeling (26-connected)
    labels = cc3d.connected_components(stroke_binary).astype(np.float64)

    # Create new dataset
    clustered_ds = xr.Dataset(
        {'Stroke_CC3d': (['time', 'y', 'x'], labels)},
        coords={'time': time_coords, 'y': y_coords, 'x': x_coords}
    )
    clustered_ds = clustered_ds.astype(np.float64)
    clustered_ds.rio.write_crs(crs, inplace=True)
    # Save to NetCDF
    clustered_ds.to_netcdf(output_nc_path)


if __name__ == "__main__":
    # Example usage
    YEAR = '2015'
    # The ‘reference_tif’ represents gridded lightning data derived from discrete lightning point observations,
    # using a spatial resolution of 0.25° and a temporal interval of 1 hour.
    # The attribute value of each grid cell represents the number of lightning strokes occurring within that cell.
    reference_tif = rf"D:/A1_lightning_stroke_tif/Stroke_count_{YEAR}/Month_01/Day_01/R{YEAR}_01_01_00.tif"
    # 'input_nc' refers to the binarized form of the “reference_tif” dataset, in which grid cells
    # are assigned binary values based on the presence or absence of lightning strokes, and is stored in NetCDF (.nc) format.
    input_nc = rf"D:/A2_lightning_stroke_Binary/Stroke_{YEAR}_Binary.nc"
    # 'output_nc' refers to the output of the CC3D clustering algorithm, representing the labeled results of connected-component analysis,
    # and is stored in NetCDF (.nc) format.
    output_nc = rf"D:/A3_CC3d_Output/Stroke_{YEAR}_CC3d.nc"

    cluster_lightning_data(input_nc, reference_tif, output_nc)