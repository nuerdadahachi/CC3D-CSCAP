# -*- coding: utf-8 -*-
import numpy as np
from geopy.distance import great_circle

"""
ST-DBSCAN implementation with adaptive spatial and temporal thresholds.

Dependencies:
    - numpy
    - geopy

This module provides the core functionality for spatial-temporal clustering 
based on adaptive distance thresholds. It is designed as a subroutine for 
larger ST-DBSCAN workflows.
"""


def computeTimeDistance(DF_data, center_point):
    """
    Compute temporal distance (in minutes) between a center point and all other points.
    """
    dtCenter = center_point['Time']
    timeDiff = np.abs(dtCenter-DF_data['Time'])

    return timeDiff.apply(lambda x: x.total_seconds() / 60).values


def computeSpatialDistance(row, coord):
    """
    Compute spatial distance (in kilometers) using great-circle distance.
    """
    coord1 = coord
    coord2 = row
    DDkm = great_circle((coord1['Latitude'], coord1['Longitude']), (coord2['Latitude'], coord2['Longitude'])).kilometers

    return DDkm


def isCorePoint(dist_S,epsR,dist_T,epsT,minPts):
    if np.sum(np.where((dist_S <= epsR) & (dist_T <= epsT), 1, 0)) >= minPts:
        return True
    return False


def adaptiveSTDBSCAN(DF_data, epsRArr, epsTArr, minPts):
    """
    Core implementation of ST-DBSCAN with adaptive spatial/temporal thresholds.

    Parameters:
        data     : DataFrame with columns ['Longitude', 'Latitude', 'Time']
        epsRArr  : Array of spatial thresholds (in km), one per point
        epsTArr  : Array of temporal thresholds (in minutes), one per point
        minPts   : Minimum number of neighbors to form a core point

    Returns:
        labels   : Cluster labels for each point (-1 means noise)
    """
    n = DF_data.shape[0]
    labels = np.full(n, -1, dtype=int)
    clusterId = 1

    for i in range(n):
        if labels[i] != -1:
            continue  # Already processed

        centerPoint = DF_data.iloc[i,:]
        timeDist = computeTimeDistance(DF_data, centerPoint)
        spatialDist = DF_data.apply(computeSpatialDistance, coord=centerPoint, axis=1).values

        if not isCorePoint(spatialDist, epsRArr[i], timeDist, epsTArr[i], minPts):
            continue

        labels[i] = clusterId
        seeds = set(np.where((spatialDist <= epsRArr[i]) &
                             (timeDist <= epsTArr[i]) &
                             (labels == -1))[0])

        while seeds:
            newPoint = seeds.pop()
            labels[newPoint] = clusterId

            newCenter = DF_data.iloc[newPoint]
            timeDistNew = computeTimeDistance(DF_data, newCenter)
            spatialDistNew = computeSpatialDistance(DF_data, newCenter)

            neighbors = np.where((spatialDistNew <= epsRArr[newPoint]) &
                                 (timeDistNew <= epsTArr[newPoint]))[0]

            if len(neighbors) >= minPts:
                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        seeds.add(neighbor)

        clusterId += 1

    return labels
