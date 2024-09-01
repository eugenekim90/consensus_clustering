#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of the consensus clustering implementation.

This script demonstrates how to use the consensus clustering module
with the provided sample dataset.
"""

import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA, VectorAssembler, Imputer
from pyspark.sql.types import NumericType, StringType
from pyspark.sql.functions import col, desc
import pandas as pd
import numpy as np

import consensus_clustering as cc
import create_radar as cr

def main():
    """Run the example consensus clustering workflow."""
    
    try:
        print("="*80)
        print("CONSENSUS CLUSTERING EXAMPLE")
        print("="*80)
        
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("ConsensusClusteringExample") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        spark.sparkContext.setLogLevel("ERROR")
        
        print("\n1. Loading and processing data...")
        # Load the sample data
        input_file = "ccdata.csv"
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Data file '{input_file}' not found!")
            
        df = spark.read.csv(input_file, header=True, inferSchema=True)
        
        # Display data summary
        print(f"   - Data loaded: {df.count()} rows")
        print(f"   - Sample columns: {df.columns[:5]}")
        
        # Prepare data for clustering
        print("\n2. Preparing data for clustering...")
        
        # Add a unique ID column if not present
        df = df.withColumnRenamed("CUST_ID", "uid")
        
        # Get the schema to filter out non-numeric columns
        schema = df.schema
        # Select only numeric columns for features
        feature_cols = [field.name for field in schema.fields 
                       if isinstance(field.dataType, NumericType) and field.name != "uid"]
        
        print(f"   - Selected {len(feature_cols)} numeric features for clustering")
        
        # Handle null values using Imputer
        print("   - Handling missing values...")
        imputer = Imputer(
            inputCols=feature_cols,
            outputCols=[f"{col}_imputed" for col in feature_cols]
        ).setStrategy("mean")
        
        df = imputer.fit(df).transform(df)
        
        # Use the imputed columns for further processing
        imputed_feature_cols = [f"{col}_imputed" for col in feature_cols]
        
        # Create vector assembler for PCA
        assembler = VectorAssembler(
            inputCols=imputed_feature_cols,
            outputCol="features",
            handleInvalid="skip"  # Skip rows with invalid values
        )
        df_assembled = assembler.transform(df)
        
        print(f"   - Assembled features for {df_assembled.count()} rows")
        
        # Apply PCA to reduce dimensionality
        pca_components = min(10, len(feature_cols))
        print(f"   - Applying PCA with {pca_components} components...")
        
        pca = PCA(k=pca_components, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df_assembled)
        df_pca = pca_model.transform(df_assembled)
        
        # Display explained variance
        explained_variance = pca_model.explainedVariance.toArray()
        cum_explained_variance = np.cumsum(explained_variance)
        print(f"   - PCA explained variance: {cum_explained_variance[-1]:.2%}")
        
        # Run consensus clustering
        print("\n3. Running consensus clustering...")
        k = 6  # Number of clusters
        iterations = 30  # Number of clustering iterations
        print(f"   - Parameters: k={k}, iterations={iterations}")
        
        start_time = time.time()
        results = cc.consensus_clustering(spark, df_pca, k, iterations, verbose=True)
        end_time = time.time()
        print(f"   - Clustering completed in {end_time - start_time:.2f} seconds")
        
        # Display clustering results summary
        cluster_counts = results.groupBy("final_cluster").count().orderBy("final_cluster")
        print("\n4. Cluster distribution:")
        cluster_counts.show()
        
        # Show average consensus strength by cluster
        print("\n5. Consensus strength by cluster:")
        results.groupBy("final_cluster") \
               .agg({"consensus_strength": "avg"}) \
               .withColumnRenamed("avg(consensus_strength)", "avg_consensus") \
               .orderBy(desc("avg_consensus")) \
               .show()
        
        # Convert to pandas for visualization
        print("\n6. Creating visualization...")
        
        # Prepare data for visualization
        # Get both the original data and the clustering results
        data_for_viz = df.join(results, on="uid", how="inner")
        
        # Select a subset of columns for visualization (uid, final_cluster, and feature columns)
        # Use feature columns with imputed values
        viz_cols = ["uid", "final_cluster", "consensus_strength"] + imputed_feature_cols
        
        # Convert to pandas for visualization
        merged_data = data_for_viz.select(viz_cols).toPandas()
        
        # Create radar charts for cluster visualization
        cr.create_radar_charts_grid(merged_data, "Cluster Characteristics")
        
        # Clean up
        spark.stop()
        print("\nExample completed successfully!")
        print("See 'cluster_visualization.png' for the visualization.")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 