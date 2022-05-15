from bandit_learn.feature_loader import FeatureLoader
import numpy as np
import pandas as pd
import dask.dataframe as dd

featureloader = FeatureLoader(
    datasetDir="./data/TSLA_2020_2022/processed_data/",
    featuresDir="./data/TSLA_2020_2022/features_test/",
    n_arms=10,
    n_features=10
)
featureloader.loadFeatures()
features = featureloader.features_df["features"].values.compute_chunk_sizes()
inference = featureloader.features_df["inference"].values.compute_chunk_sizes()
confidence = featureloader.features_df["confidence"].values.compute_chunk_sizes()
for i in range(10):
    print(features[0].compute()[i])
    print(inference[0].compute()[i])
    print(confidence[0].compute()[i])
