#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consensus Clustering implementation using Apache Spark.

This module provides a function to perform consensus clustering using
K-means algorithm with multiple iterations to improve clustering stability.
"""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def consensus_clustering(spark, df_pca, k, iterations, verbose=True):
    """
    Perform consensus clustering using K-means algorithm with multiple iterations.
    
    This function:
    1. Runs K-means clustering multiple times with different random seeds
    2. Resolves label inconsistencies using the Hungarian algorithm
    3. Determines final cluster labels through majority voting
    
    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Spark session object.
    df_pca : pyspark.sql.DataFrame
        Input DataFrame containing PCA-transformed features in 'pca_features' column
        and a unique identifier column 'uid'.
    k : int
        Number of clusters for K-means.
    iterations : int
        Number of iterations for K-means clustering to generate consensus clusters.
    verbose : bool, optional
        Whether to print progress information, default is True.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame with columns 'uid' and 'final_cluster', representing the unique 
        identifier and the consensus cluster label for each data point.
    """

    if verbose:
        print("----- Starting Consensus Clustering -----")
        start_time = time.time()

    # 1. Run K-means Clustering Multiple Times:
    cluster_results = []
    silhouette_scores = []
    
    evaluator = ClusteringEvaluator(
        predictionCol='prediction', 
        featuresCol='pca_features', 
        metricName='silhouette', 
        distanceMeasure='squaredEuclidean'
    )
    
    # Cache the DataFrame to speed up iterative processing
    df_pca.cache()

    for i in range(iterations):
        # Initialize and fit a K-means model with a different seed for each iteration
        kmeans = KMeans(k=k, seed=i, featuresCol='pca_features', predictionCol='prediction', maxIter=100)
        model = kmeans.fit(df_pca)
        transformed = model.transform(df_pca)
        
        # Calculate silhouette score for this iteration's clustering
        silhouette = evaluator.evaluate(transformed)
        silhouette_scores.append(silhouette)

        # Store clustering results with unique prediction column names for each iteration
        result = transformed.select('uid', 'prediction')
        cluster_results.append(result.withColumnRenamed('prediction', f'prediction_{i}'))
        if verbose:
            print(f'- K-means iteration {i+1}/{iterations} complete (Silhouette: {silhouette:.4f})')

    if verbose:
        avg_silhouette = sum(silhouette_scores) / len(silhouette_scores)
        print(f"- Average Silhouette Score: {avg_silhouette:.4f}")
        print("- Combining clustering results...")

    # 2. Combine Clustering Results:
    # Join all DataFrames containing clustering results based on the unique identifier ('uid')
    joined_df = cluster_results[0]
    for result in cluster_results[1:]:
        joined_df = joined_df.join(result, on='uid', how='inner')

    # Collect all unique identifiers for subsequent processing
    uids = joined_df.select('uid').rdd.flatMap(lambda x: x).collect()

    if verbose:
        print("- Resolving label inconsistencies...")

    # 3. Resolve Label Inconsistency (Hungarian Algorithm):
    base_labels = joined_df.select('prediction_0').rdd.flatMap(lambda x: x).collect()
    resolved_labels = [base_labels]  # Initialize with base labels

    for i in range(1, iterations):
        new_labels = joined_df.select(f'prediction_{i}').rdd.flatMap(lambda x: x).collect()
        
        # Create a cost matrix where each entry represents the number of mismatches between labels
        cost_matrix = np.zeros((k, k))
        for m in range(k):
            for n in range(k):
                cost_matrix[m, n] = np.sum((np.array(base_labels) == m) != (np.array(new_labels) == n))

        # Apply the Hungarian algorithm to find the optimal mapping between labels
        # to minimize the total cost (number of mismatches)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {col: row for row, col in zip(row_ind, col_ind)}
        mapped_labels = np.vectorize(mapping.get)(new_labels)
        resolved_labels.append(mapped_labels)

    resolved_labels = np.array(resolved_labels).T 

    if verbose:
        print("- Determining final consensus clusters...")

    # 4. Final Consensus Clustering:
    # Determine the final cluster label for each data point through majority voting
    final_cluster_labels = [int(np.bincount(row).argmax()) for row in resolved_labels]
    
    # Calculate consensus strength (percentage of iterations agreeing with final label)
    consensus_strengths = []
    for row in resolved_labels:
        final_label = np.bincount(row).argmax()
        strength = np.mean(row == final_label)
        consensus_strengths.append(float(strength))
    
    # Create a DataFrame containing the unique identifier ('uid'), final cluster label,
    # and consensus strength
    final_cluster_df = spark.createDataFrame(
        [(uid, label, strength) 
         for uid, label, strength in zip(uids, final_cluster_labels, consensus_strengths)],
        ['uid', 'final_cluster', 'consensus_strength']
    )
    
    # Calculate average consensus strength per cluster
    if verbose:
        for cluster in range(k):
            cluster_strength = final_cluster_df.filter(f"final_cluster = {cluster}") \
                               .select("consensus_strength").rdd.map(lambda x: x[0]).mean()
            print(f"- Cluster {cluster} consensus strength: {cluster_strength:.4f}")
        
        # Calculate overall consensus strength
        overall_strength = final_cluster_df.select("consensus_strength").rdd.map(lambda x: x[0]).mean()
        print(f"- Overall consensus strength: {overall_strength:.4f}")
        
        end_time = time.time()
        print(f"----- Consensus Clustering Complete (Runtime: {end_time - start_time:.2f}s) -----")
    
    # Uncache the DataFrame 
    df_pca.unpersist()
    
    return final_cluster_df 