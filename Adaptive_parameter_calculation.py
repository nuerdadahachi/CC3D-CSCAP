# -*- coding: utf-8 -*-
"""
Adaptive spatial and temporal threshold estimation module for Fine_scale_identification.

Author: [Manxing Shi]
Date: [August 7, 2025]
"""

import numpy as np
from geopy.distance import great_circle
from datetime import timedelta
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import math


def subset_by_time(index_center, df, temporal_threshold):
    """
    Subset lightning events within a temporal window centered at a given event.

    Parameters:
        index_center (int): Index of the central event in the dataframe.
        df (DataFrame): Lightning event dataframe with a '时间' (timestamp) column.
        temporal_threshold (float): Temporal window in minutes.

    Returns:
        df_subset (DataFrame): Events within ± temporal_threshold of the center.
        center_point (Series): The central lightning event.
    """
    center_point = df.iloc[index_center]
    min_time = center_point['Time'] - timedelta(minutes=temporal_threshold)
    max_time = center_point['Time'] + timedelta(minutes=temporal_threshold)
    df_subset = df[(df['Time'] >= min_time) & (df['Time'] <= max_time)]
    return df_subset, center_point


def compute_time_distances(df_subset, center_point):
    """
    Compute temporal distances (in minutes) from center to all neighbors.

    Returns:
        np.ndarray: Sorted array of time distances.
    """
    time_deltas = abs(center_point['Time'] - df_subset['Time'])
    minutes = time_deltas.apply(lambda x: x.total_seconds() / 60)
    return np.sort(minutes.values)


def haversine_distance_dd(coord1, coord2):
    """Compute great-circle distance (in kilometers) between two lon-lat points."""
    return great_circle((coord1[1], coord1[0]), (coord2[1], coord2[0])).kilometers


def compute_spatial_distances(df_subset, center_point, spatial_threshold):
    """
    Compute spatial distances (km) within a spatial threshold using BallTree.

    Returns:
        np.ndarray: Distance values to nearby points within the spatial threshold.
    """
    coords = df_subset[['Longitude', 'Latitude']].values
    tree = BallTree(coords, metric=haversine_distance_dd)
    query_point = np.array([center_point[['Longitude', 'Latitude']].values])
    _, distances = tree.query_radius(query_point, r=spatial_threshold,
                                     return_distance=True, sort_results=True)
    return distances[0]


def compute_temporal_spatial_distributions(df, temporal_threshold=30, spatial_threshold=50):
    """
    For each lightning event, compute local temporal and spatial distributions.

    Returns:
        list of np.ndarray: Spatial distances
        list of np.ndarray: Temporal distances
    """
    spatial_dists, temporal_dists = [], []

    for i in range(df.shape[0]):
        df_sub, center = subset_by_time(i, df, temporal_threshold)
        if df_sub.shape[0] < 2:
            spatial_dists.append(np.array([]))
            temporal_dists.append(np.array([]))
        else:
            spatial_dists.append(compute_spatial_distances(df_sub, center, spatial_threshold))
            temporal_dists.append(compute_time_distances(df_sub, center))

    return spatial_dists, temporal_dists


def optimal_inflection_point(x, y):
    """
    Find the inflection point on a curve where y = f(x) crosses y = 1 - x.

    Returns:
        tuple: (ROC-like ratio, x-coordinate of intersection in original scale)
    """
    scaler_x = MinMaxScaler()
    x_scaled = MinMaxScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten()
    y_scaled = MinMaxScaler().fit_transform(np.array(y).reshape(-1, 1)).flatten()

    f_interp = interp1d(x_scaled, y_scaled, kind='linear')

    def equation(x_val):
        return f_interp(x_val) - (1 - x_val)

    x_guess =  np.array([0.5])
    x_intersect = fsolve(equation, x_guess)[0]
    y_intersect = 1 - x_intersect

    roc_ratio = np.linalg.norm([x_intersect, y_intersect] - np.array([0, 1])) / np.sqrt(2)
    x_original = scaler_x.inverse_transform([[x_intersect]])[0][0]

    return roc_ratio, x_original


def adaptive_threshold(values, default=15):
    """
    Estimate an adaptive threshold based on ROC-like inflection analysis.

    Returns:
        float: Optimal threshold or default value.
    """
    if len(values) <= 3 or max(values) <= default or np.any(values[:3] > default):
        return default

    rate_rocs, thresholds = [], []
    while len(values) > 2 and max(values) > default:
        rate, thresh = optimal_inflection_point(values, range(1, len(values) + 1))
        rate_rocs.append(rate)
        thresholds.append(thresh)
        values = values[:-math.ceil(len(values) * 0.1)]

    if len(rate_rocs) == 0:
        return default

    min_index = np.argmin(rate_rocs)
    best_thresh = thresholds[min_index]

    return best_thresh if rate_rocs[min_index] <= 0.5 else default


def compute_adaptive_RT(df):
    """
    Compute adaptive spatial and temporal thresholds for each lightning event.

    Returns:
        list: Adaptive spatial thresholds (R)
        list: Adaptive temporal thresholds (T)
    """
    spatial_results, temporal_results = compute_temporal_spatial_distributions(df)
    adaptive_Rs, adaptive_Ts = [], []

    for s_dist, t_dist in zip(spatial_results, temporal_results):
        R = adaptive_threshold(s_dist[s_dist != 0])
        T = adaptive_threshold(t_dist[t_dist != 0])
        adaptive_Rs.append(R)
        adaptive_Ts.append(T)

    return adaptive_Rs, adaptive_Ts
