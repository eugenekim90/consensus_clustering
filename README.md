# Consensus Clustering with Apache Spark

A robust implementation of consensus clustering using Apache Spark, designed for large-scale data analysis and clustering stability assessment.

## Overview

This project implements consensus clustering, a technique that improves clustering stability by aggregating results from multiple clustering runs. The implementation:

- Uses Apache Spark for distributed processing
- Handles label inconsistencies using the Hungarian algorithm
- Performs majority voting to determine final cluster assignments
- Calculates consensus strength metrics to assess clustering stability
- Includes visualization tools for analyzing cluster characteristics

## Components

### 1. `consensus_clustering.py`
Core implementation of the consensus clustering algorithm with the following features:
- Multiple K-means clustering iterations with different random seeds
- Label matching using the Hungarian algorithm to resolve label switching
- Aggregation of results using majority voting
- Silhouette score calculation for clustering quality evaluation
- Consensus strength metrics to measure clustering stability

### 2. `create_radar.py`
Visualization tool to create radar charts that display the characteristic features of each cluster:
- Normalizes feature values
- Creates a grid of radar charts for comparing clusters
- Displays mean feature values for each cluster
- Shows cluster size and percentage information
- Automatically saves visualization to image file

### 3. `example.py`
Example script demonstrating the full workflow:
- Data loading and preprocessing
- Feature engineering with PCA
- Running consensus clustering
- Analyzing cluster results
- Generating visualizations

### 4. `ccdata.csv`
Sample dataset for clustering (contains feature data for consensus clustering)

## Usage

```python
from pyspark.sql import SparkSession
import consensus_clustering as cc
import create_radar as cr
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ConsensusClusteringExample") \
    .getOrCreate()

# Load and prepare your data
df = spark.read.csv("ccdata.csv", header=True, inferSchema=True)

# Preprocess and prepare features
# (See example.py for full preprocessing workflow)

# Perform consensus clustering
k = 6  # Number of clusters
iterations = 30  # Number of clustering iterations
results = cc.consensus_clustering(spark, df_pca, k, iterations)

# Analyze clustering results
cluster_counts = results.groupBy("final_cluster").count().orderBy("final_cluster")
print("Cluster distribution:")
cluster_counts.show()

# Analyze consensus strength
results.groupBy("final_cluster") \
       .agg({"consensus_strength": "avg"}) \
       .withColumnRenamed("avg(consensus_strength)", "avg_consensus") \
       .orderBy("final_cluster") \
       .show()

# Prepare data for visualization
data_for_viz = df.join(results, on="uid", how="inner")
viz_cols = ["uid", "final_cluster", "consensus_strength"] + feature_cols
merged_data = data_for_viz.select(viz_cols).toPandas()

# Create radar charts for cluster visualization
cr.create_radar_charts_grid(merged_data, "Cluster Characteristics")
```

## Requirements

- Apache Spark >= 3.0.0
- Python 3.x
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.2.0
- scikit-learn >= 0.23.0
- pandas >= 1.0.0

## Implementation Details

The consensus clustering algorithm follows these steps:

1. Run K-means clustering multiple times with different random initializations
2. Calculate silhouette score for each clustering iteration to assess quality
3. Resolve label inconsistencies using the Hungarian algorithm
4. Determine final cluster labels through majority voting
5. Calculate consensus strength metrics for each data point and cluster
6. Return a DataFrame with unique identifiers, consensus cluster labels, and consensus strength

## Visualization

The radar chart visualization shows the following information for each cluster:
- Mean normalized feature values for each cluster
- Cluster size (number of data points)
- Cluster percentage (proportion of the total dataset)
- Feature values displayed on each axis

The visualization is automatically saved as `cluster_visualization.png` in the current directory.
