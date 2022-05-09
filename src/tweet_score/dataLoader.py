# load tweets data and use model to predict score
import pandas as pd
import os
from .model import TweetAnalyzer
from dask import dataframe as dd
from datetime import datetime

# pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("display.width", 1000)


class DataLoader(object):
    def __init__(self, readDir=None, resultDir=None, analyzer=None, *args):
        super(DataLoader, self).__init__(*args)
        self.readDir = readDir
        self.resultDir = resultDir
        self.analyzer: TweetAnalyzer = analyzer
        self.read_df = None
        self.result_df = None

    def readTweets(self, n=-1):
        assert self.readDir != None
        if n != -1:
            print("[NOTICE] n should set -1 if you want to all data (in readManyCsv)")
        print(" Reading CSVs [V]")
        filepaths = [
            self.readDir + f
            for f in os.listdir(self.readDir)
            if f.endswith(".csv")
        ]
        self.read_df = dd.read_csv(
            filepaths,
            parse_dates={"Date": ["created_at"]},
            dtype={
                "user": "object",
                "symbols": "object",
                "source": "object",
                "mentioned_users": "object",
                "entities": "object",
                "owned_symbols": "object",
                "likes": "object",
                "reshares": "object",
                "conversation": "object",
                "reshare_message": "object",
                "owned_symbols": "object",
                "links": "object",
            }
        )
        self.read_df = self.read_df if n == -1 else self.read_df.head(n)


    def labelTweets(self):
        print(" labeling data [V]")
        # bodyList = self.read_df["body"]
        self.result_df = dd.concat(
            [*self.analyzer.batchTokenize(self.read_df[["id", "body"]], needProcessed=True)]
        )

    def writeResults(self):
        print(" Writing Results [V]")
        if not os.path.exists(self.resultDir):
            os.makedirs(self.resultDir)
        self.result_df.to_csv(self.resultDir+"TSLA_2020_2022_*.csv")

    def extendResults(self, cols: list):
        cols = [
            c for c in self.read_df.columns
            if c in cols and c not in self.result_df.columns
        ]
        # print(self.read_df.columns)
        # print(self.result_df.columns)
        self.result_df = dd.concat([self.result_df, self.read_df[cols].set_index("id")], axis=1)


# def main():
#     print("=======Start========")
#     analyzer = TweetAnalyzer()
    
#     TweetLoader = DataLoader(
#         readDir="./data/TSLA_2020_2022/",
#         resultDir="./data/TSLA_2020_2022/labeled_data/",
#         analyzer=analyzer
#     )

#     TweetLoader.readTweets(n=1000)
#     TweetLoader.labelTweets()
#     TweetLoader.writeResults()
#     TweetLoader.extendResults("created_at,user,source,symbols,".split(","))
#     print(TweetLoader.result_df.head())
#     print("======= End ========")
#     print("Data shape: ", TweetLoader.result_df.shape)


# if __name__ == "__main__":
#     start_time = datetime.now()
#     main()
#     end_time = datetime.now()
#     print("Duration: {}".format(end_time - start_time))
