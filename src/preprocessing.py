from tweet_score.dataLoader import DataLoader
from tweet_score.model import TweetAnalyzer
from datetime import datetime
from helper import Only_dict
# Selected Column
import os
import pandas as pd


class Preprocessing(object):
    def __init__(self, read_dir, result_dir, *args):
        super(Preprocessing, self).__init__(*args)
        self.df = None
        self.readDir = read_dir
        self.resultDir = result_dir

    def extendJsonCols(self, wantDrop=True):
        '''
        處理Json格式的columns
        '''
        jsonCols = [
            col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in jsonCols:
            print(jsonCols)
            self.df[col] = self.df[col].fillna('{}')
            new_df = pd.json_normalize(self.df[col].apply(
                Only_dict).tolist(), sep='.').add_prefix(col+'.')
            self.df = self.df.join(new_df)
            self.df.drop(columns=[col], inplace=wantDrop)

    def printInfo(self):
        for c in self.df.columns:
            print(c)
        print(self.df.shape)
        print(self.df.info())

    def handleMissingVal():
        '''
        處理missing value
        '''
        pass

    def modifyData(self, dropCols=[]):
        '''
        處理drop資料、改變資料的type
        '''
        self.df.drop(
            columns=dropCols, inplace=True)
        self.df.label = self.df.label.astype('string')
        self.df.sentences = self.df.sentences.astype('string')

    def ReadLabelData(self):
        filepaths = [
            self.readDir + f
            for f in os.listdir(self.readDir)
            if f.endswith(".csv")
        ]
        self.df = pd.concat([pd.read_csv(path)for path in filepaths])

    def writeResults(self):
        '''
        寫處理過的資料到檔案
        '''
        print(" Writing Results [V]")
        if not os.path.exists(self.resultDir):
            os.makedirs(self.resultDir)
        self.df.to_csv(self.resultDir+"TSLA_2020_2022.csv")

# - Id
# - Created_at
# - User
#   - official(bool)
#   - join_date
#   - followers
#   - following
#   - ideas
#   - watchlist_stocks_count
#   - like_count
# - Likes
#   - total
# - Conversation -> bool
# - Reshare_message
#   - reshared_count
# - Reshares
#   - reshared_count


def tweet():
    print("=======Start========")
    analyzer = TweetAnalyzer()

    TweetLoader = DataLoader(
        readDir="./data/TSLA_2020_2022/",
        resultDir="./data/TSLA_2020_2022/labeled_data/",
        analyzer=analyzer,
        n_partition=1000
    )
    TweetLoader.readTweets()
    TweetLoader.labelTweets()
    TweetLoader.extendResults(
        "id,created_at,user,likes,reshares,conversation,reshare_message".split(','))
    TweetLoader.writeResults()
    print(TweetLoader.result_df.head())
    print("======= End ========")
    print("Data shape: ", TweetLoader.result_df.shape)


def testPreprocessing():
    pre = Preprocessing(read_dir='./data/TSLA_2020_2022/labeled_data/',
                        result_dir='./data/TSLA_2020_2022/processed_data/')
    pre.ReadLabelData()
    pre.modifyData(dropCols=['reshares', 'reshare_message', 'Unnamed: 0'])
    pre.extendJsonCols()
    pre.printInfo()
    pre.writeResults()


if __name__ == '__main__':
    start_time = datetime.now()
    # tweet()
    testPreprocessing()
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))
