#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Radar chart visualization for cluster analysis.

This module provides functionality to create radar charts that visualize
the characteristic features of each cluster identified by consensus clustering.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def create_radar_charts_grid(data, title):
    """
    Create a grid of radar charts showing feature characteristics for each cluster.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing feature columns and a 'final_cluster' column 
        with cluster assignments
    title : str
        The title to display above the grid of charts
        
    Returns
    -------
    None
        Displays the radar charts using matplotlib
        
    Notes
    -----
    This function:
    1. Extracts features and normalizes their values
    2. Creates a polar plot for each cluster
    3. Displays the mean feature values for each cluster as a radar chart
    """
    # Extract features and clusters (exclude 'uid' and 'final_cluster' columns)
    features = [col for col in data.columns 
              if col != 'final_cluster' and col != 'uid'
              and pd.api.types.is_numeric_dtype(data[col])]
    
    # If we have more than 8 features, limit to first 8 for readability
    if len(features) > 8:
        features = features[:8]
    
    # Get cluster information
    clusters = data['final_cluster'].unique()
    num_clusters = len(clusters)
    total_points = len(data)
    
    # Calculate cluster sizes and percentages
    cluster_stats = {}
    for cluster in clusters:
        size = len(data[data['final_cluster'] == cluster])
        percentage = (size / total_points) * 100
        cluster_stats[cluster] = {'size': size, 'percentage': percentage}
    
    print(f"Creating radar charts for {num_clusters} clusters with {len(features)} features")

    # Normalize the feature values to [0,1] range
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[features])
    
    # Convert normalized data back to DataFrame for easier manipulation
    normalized_df = pd.DataFrame(normalized_data, columns=features)
    normalized_df['final_cluster'] = data['final_cluster'].values

    # Prepare angles for radar chart (evenly spaced around the circle)
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Calculate grid layout (3 charts per row)
    rows = (num_clusters + 2) // 3

    # Create subplots grid
    fig, axs = plt.subplots(rows, 3, figsize=(18, 6 * rows), subplot_kw=dict(polar=True))
    if rows == 1 and num_clusters <= 3:
        axs = np.array([axs])  # Ensure it's an array for consistent indexing
    axs = axs.flatten()

    # Create radar chart for each cluster
    for i, cluster in enumerate(sorted(clusters)):
        # Get data for this cluster and calculate mean values
        cluster_data = normalized_df[normalized_df['final_cluster'] == cluster][features]
        values = cluster_data.mean().tolist()
        values += values[:1]  # Close the circle

        # Get cluster statistics
        stats = cluster_stats[cluster]
        
        # Plot the radar chart
        axs[i].fill(angles, values, alpha=0.25, color='b')
        axs[i].plot(angles, values, color='b', linewidth=2)
        axs[i].set_ylim(0, 1)
        axs[i].set_yticklabels([])
        axs[i].set_xticks(angles[:-1])
        axs[i].set_xticklabels(features, fontsize=9)

        # Add value labels
        for j in range(num_vars):
            axs[i].text(angles[j], values[j], f'{values[j]:.2f}', 
                        horizontalalignment='center', size=8, 
                        color='black', weight='semibold')

        # Add cluster information in the title
        axs[i].set_title(f"Cluster {cluster}: {stats['size']} points ({stats['percentage']:.1f}%)",
                        fontsize=12, fontweight='bold', pad=15)

    # Remove unused subplots if any
    for i in range(num_clusters, len(axs)):
        fig.delaxes(axs[i])

    # Add main title and layout adjustments
    plt.suptitle(f'{title}', size=20, color='blue', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure to a file
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'cluster_visualization.png'")
    
    # Display the figure
    plt.show()