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
CC3D-CSCAP/
├── Coarse_scale_identification.py # Coarse-scale 3D CC3D clustering
├── Fine_scale_identification.py # Adaptive ST-DBSCAN clustering
├── STDBSCAN.py # Core ST-DBSCAN implementation
├── Adaptive_parameter_calculation.py # Adaptive threshold estimation
├── requirements.txt # Python package dependencies
├── README.md # Project description
├── examples/
│ ├── input_sample.csv
│ └── output_sample.csv

---

## Requirements

- Python 3.7+
- numpy
- pandas
- geopy
- xarray
- rioxarray
- cc3d

Install dependencies using:

```bash
pip install -r requirements.txt
Usage
1. Coarse-Scale Cluster Identification
python Coarse_scale_identification.py
Input: Hourly binary raster lightning data (.nc) with spatial resolution of 0.25°
Output: 3D labeled connected-component clusters (.nc)
2. Fine-Scale Cluster Refinement
python Fine_scale_identification.py
Input: CSV files containing stroke data grouped by CC3D cluster
Output: CSV files labeled with refined storm system IDs
License
MIT License © 2025
Contact
For questions or contributions, please contact the repository maintainer via GitHub issues or pull requests.
