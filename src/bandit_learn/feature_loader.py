import numpy as np
import os
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler, MinMaxScaler

class FeatureLoader():
    def __init__(
        self,
        datasetDir: str,
        featuresDir: str,
        n_features: int,
        n_arms: int,
    ):
        self.datasetDir = datasetDir
        self.featuresDir = featuresDir
        self.n_features = n_features
        self.n_arms = n_arms
        
    def scaleData(self, standard_scale_columns, min_max_scale_columns):
        self.df = dd.read_csv(self.datasetDir)
        
        self.df["user.official"] = self.df["user.official"].astype("int64")
        self.df["reshare"] = self.df["reshare"].astype("int64")
        
        # standard_scale_columns = [
        #     "user.official",
        #     "user.followers",
        #     "user.following",
        #     "user.ideas",
        #     "user.watchlist_stocks_count",
        #     "user.like_count",
        #     "likes.total",
        #     "conversation.replies",
        #     "reshare",
        # ]
        standard_scaler = StandardScaler()
        scaled_df = dd.DataFrame()
        for col in standard_scale_columns:
            scaled_df[col] = standard_scaler.fit_transform(self.df[col])
        
        # min_max_scale_columns = ["user.join_date"]
        minmaxscaler = MinMaxScaler()
        for col in min_max_scale_columns:
            scaled_df[col] = minmaxscaler.fit_transform(self.df[col])

    def createFeatures(self):
        pass

    def saveFeatures(self):
        pass
        
        

