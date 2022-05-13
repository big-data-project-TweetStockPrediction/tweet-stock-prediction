import numpy as np
import os
import dask.dataframe as dd
import pandas as pd
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
        self.df = None
        self.scaled_df = None

    def loadData(self):
        self.df = dd.read_csv(self.datasetDir)

    def modifyDataType(self, cols: list, dataType: str):
        # self.df["user.official"] = self.df["user.official"].astype("int64")
        # self.df["reshare"] = self.df["reshare"].astype("int64")
        for col in cols:
            self.df[col] = self.df[col].astype(dataType)
        
    def scaleData(self,
        standard_scale_columns,
        min_max_scale_columns,
        other_columns,
    ):
        self.scaled_df = dd.DataFrame()

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
        for col in standard_scale_columns:
            self.scaled_df[col] = standard_scaler.fit_transform(self.df[col])
        
        # min_max_scale_columns = ["user.join_date"]
        minmaxscaler = MinMaxScaler()
        for col in min_max_scale_columns:
            self.scaled_df[col] = minmaxscaler.fit_transform(self.df[col])

        # other_columns = ["Date", "label", "score"]
        for col in other_columns:
            self.scaled_df[col] = self.df[col]

    def createFeatures(self):
        start_date = pd.to_datetime("03-01-2020")
        end_date = pd.to_datetime("03-02-2022")
        self.features_df =  dd.DataFrame()
        for date in pd.date_range(start=start_date, end=end_date, dtype="datetime64[ns, UTC]"):
            date_df: dd.DataFrame = self.scaled_df[self.scaled_df["Date"] == date]
            date_df = date_df.loc[:, date_df.columns != "Date"]
            
            rows: np.ndarray = date_df.to_numpy()
            rand_indices = np.random.randint(
                rows.shape[0],
                size=self.n_arms - rows.shape[0] % self.n_arms
            )
            rand_features = rows[rand_indices, :]
            rows = np.concatenate((rows, rand_features), axis=0)
            indies = np.arange(rows.shape[0]).reshape((-1, self.n_arms))
            np.random.shuffle(indies)
            
            arr = np.array(
                [
                    [
                        date,
                        rows[idx, :self.n_features],
                        rows[idx, self.n_features],
                        rows[idx, self.n_features + 1],
                    ]
                    for idx in indies
                ]
            )
            columns = ["date", "features", "inference", "confidence"]
            meta = pd.DataFrame(
                dict(
                    date=pd.Series(dtype='datetime64[ns, UTC]'),
                    features=pd.Series(dtype='float64'),
                    inference=pd.Series(dtype='string'),
                    confidence=pd.Series(dtype='float64'),
                )
            )
            features_df = dd.from_array(arr, columns=columns, meta=meta)
            self.features_df = dd.concat([self.features_df, features_df])

    def writeFeatures(self):
        print(" Writing Features [V]")
        if not os.path.exists(self.featuresDir):
            os.makedirs(self.featuresDir)
        self.features_df.to_csv(
            os.path.join(self.featuresDir+"TSLA_2020_2022_*.csv")
        )

    def loadFeatures(self):
        self.features_df = dd.read_csv(self.featuresDir)

