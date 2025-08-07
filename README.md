# CC3D-CSCAP: A Multi-Scale Spatiotemporal Clustering Algorithm for Thunderstorm System Identification

**CC3D-CSCAP** is a two-stage clustering framework designed to identify thunderstorm systems from large-scale lightning stroke datasets (on the order of tens of millions of points). It incorporates multi-scale spatial-temporal proximity constraints to balance computational efficiency and clustering precision.

## Overview

The algorithm consists of two main stages:

### 1. Coarse-Scale Clustering (CC3D)

- **File**: `Coarse_scale_identification.py`
- **Method**: A 3D connected component labeling (CC3D) algorithm is applied to binary rasterized lightning stroke data.
- **Purpose**: Groups neighboring lightning cells into coarse-scale clusters using spatial adjacency in 3D (latitude, longitude, and time).
- **Advantage**: Reduces the search space and computational burden for fine-scale clustering by preprocessing and filtering sparse regions.

### 2. Fine-Scale Clustering (CSCAP: Cluster Splitting by Connectivity and Adaptive Parameters)

- **File**: `Fine_scale_identification.py`
- **Method**: An adaptive parameter Spatiotemporal DBSCAN (ST-DBSCAN) algorithm that estimates optimal temporal and spatial thresholds locally for each coarse cluster.
- **Purpose**: Refines cluster boundaries and identifies individual thunderstorm systems with high temporal and spatial resolution.
- **Feature**: The algorithm automatically adjusts neighborhood thresholds according to local lightning density and distribution characteristics.

---

## Applications

- Thunderstorm tracking and climatology
- Convective system detection
- Lightning hazard mapping
- Data preprocessing for nowcasting models

---

## Repository Structure

