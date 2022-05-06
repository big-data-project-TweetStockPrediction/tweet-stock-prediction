# load tweets data and use model to predict score
import pandas as pd
import os
from model import TweetAnalyzer
from dask import dataframe as dd
from datetime import datetime

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
Analyzer = TweetAnalyzer()


class DataLoader(object):
    def __init__(self, readDir=None, writeDir=None, *args):
        super(DataLoader, self).__init__(*args)
        self.readDir = readDir
        self.writeDir = writeDir

    def readManyCsv(self, n=-1):
        assert self.readDir != None
        if n != -1:
            print("[NOTICE] n should set -1 if you want to all data (in readManyCsv)")
        print(" Reading CSVs [V]")
        filepaths = [self.readDir+f for f in os.listdir(
            self.readDir) if f.endswith('.csv')]
        df = dd.read_csv(filepaths, parse_dates={'Date': ['created_at']},
                         dtype={'owned_symbols': 'object',
                                'reshare_message': 'object',
                                'reshares': 'object'})
        df = df if n == -1 else df.head(n)
        return df

    def labelTweets(self, data, writeOpt=False):
        print(" labeling data [V]")
        bodyList = data['body'].tolist()
        scoreResult = Analyzer.tokenize(bodyList, needProcessed=True)
        scoreDict = {k: [dic[k] for dic in scoreResult]
                     for k in scoreResult[0]}
        scoreDict['sentence'] = bodyList
        data = dd.from_pandas(pd.DataFrame(data=scoreDict), npartitions=2)
        print(" Writing Results [V]")
        if writeOpt:
            if not os.path.exists(self.writeDir):
                os.makedirs(self.writeDir)
            data.to_csv(self.writeDir+'TSLA_2020_2022_*.csv')
        return data


def main():
    print("=======Start========")
    TweetLoader = DataLoader(readDir="./data/TSLA_2020_2022/",
                             writeDir="./data/TSLA_2020_2022/labeled_data/")

    df = TweetLoader.readManyCsv(n=1000)
    TweetLoader.labelTweets(data=df, writeOpt=True)
    print("======= End ========")
    print("Data shape: ", df.shape)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
