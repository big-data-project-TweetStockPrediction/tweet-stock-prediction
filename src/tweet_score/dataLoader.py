# load tweets data and use model to predict score
# import necessary libraries
import pandas as pd
import os
from model import TweetAnalyzer
import pprint
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def _readCsv(filepaths):
    return pd.read_csv(filepaths, index_col=False)


def readManyCsv(targetDir: str):
    filepaths = [targetDir+f for f in os.listdir(
        targetDir) if f.endswith('.csv')]
    df = pd.concat(map(_readCsv, filepaths), ignore_index=True)
    print(df.columns)
    print(df.head())
    print(df.info())
    df.head(100).to_csv("TSLA_2020_2022_100.csv")


if __name__ == '__main__':
    # readManyCsv("./data/TSLA_2020_2022/")\
    df = pd.read_csv("./TSLA_2020_2022_100.csv", index_col=False)
    tweets = df['body'].to_list()
    Analyzer = TweetAnalyzer()
    res = Analyzer.tokenize(tweets, needProcessed=True)
    pprint.pprint(res)
