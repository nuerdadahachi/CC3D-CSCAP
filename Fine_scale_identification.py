# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from Adaptive_parameter_calculation import compute_adaptive_RT
from STDBSCAN import adaptiveSTDBSCAN


def processClusterGroup(df_group: pd.DataFrame, minPts: int = 2) -> pd.DataFrame:
    """
    Process a single group of clustered points using Adaptive ST-DBSCAN.

    Parameters:
        df_group : A pandas DataFrame for one cluster group (same Cluster_num)
        minPts   : Minimum number of points to form a core point

    Returns:
        DataFrame with additional 'cluster' column
    """
    df_group = df_group.reset_index(drop=True)
    df_group['Time'] = pd.to_datetime(df_group['Time'])
    df_group = df_group.sort_values('Time').reset_index(drop=True)

    epsRArr, epsTArr = compute_adaptive_RT(df_group)
    epsRArr = np.array(epsRArr)
    epsTArr = np.array(epsTArr)

    labels = adaptiveSTDBSCAN(df_group, epsRArr, epsTArr, minPts)
    df_group['cluster'] = labels

    return df_group


def processCsvFile(file_path: str, output_dir: str, minPts: int = 2) -> None:
    """
    Process a single CSV file containing lightning cluster data.

    Parameters:
        file_path  : Path to the input CSV file
        output_dir : Directory to store the processed result
        minPts     : DBSCAN minPts parameter
    """
    df_all = pd.read_csv(file_path)
    cluster_ids = np.unique(df_all['Cluster_num'].values)
    df_result = pd.DataFrame()

    for cid in cluster_ids:
        df_cluster = df_all[df_all['Cluster_num'] == cid]
        processed_cluster = processClusterGroup(df_cluster, minPts)
        df_result = pd.concat([df_result, processed_cluster], ignore_index=True)

    output_path = os.path.join(output_dir, 'CSCAP_' + os.path.basename(file_path))
    df_result.to_csv(output_path, index=False, encoding='utf_8_sig')

    # Optionally remove the original file
    os.remove(file_path)


def main():
    year = '2013'
    input_dir = f'D:/Y1315_CC3d_Stroke/CC3d_{year}/'
    output_dir = f'D:/Clustering_results/Y{year}/'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            print(f'Processing file: {filename}')
            try:
                processCsvFile(file_path, output_dir)
            except Exception as e:
                print(f'[ERROR] Failed to process {filename}: {e}')


if __name__ == '__main__':
    main()