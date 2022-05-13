import re
from turtle import st

import emoji
from tweet_score.dataLoader import DataLoader
from tweet_score.model import TweetAnalyzer
from bandit_learn.feature_loader import FeatureLoader
from datetime import datetime
from helper import Only_dict
# Selected Column
import os
import pandas as pd


class Preprocessing(object):
    def __init__(self, read_dir, result_dir, fileName, *args):
        super(Preprocessing, self).__init__(*args)
        self.df = None
        self.readDir = read_dir
        self.resultDir = result_dir
        self.fileName = fileName

    def extendJsonCols(self, wantDrop=True):
        '''
        處理Json格式的columns
        '''
        jsonCols = [
            col for col in self.df.columns
            if self.df[col].dtype == 'object'
        ]
        for col in jsonCols:
            # print(jsonCols)
            self.df[col] = self.df[col].fillna('{}')
            new_df = pd.json_normalize(
                self.df[col].apply(Only_dict).tolist(),
                sep='.',
            ).add_prefix(col + '.')
            self.df = self.df.join(new_df)
            self.df.drop(columns=[col], inplace=wantDrop)

    def printInfo(self):
        for c in self.df.columns:
            print(c)
        print(self.df.shape)
        print(self.df.info())

    def process_text(self, cols: list):
        # remove URLs
        # print(texts)
        def process(texts):
            texts = re.sub(r'https?://\S+', "", texts)
            texts = re.sub(r'www.\S+', "", texts)
            # remove '
            texts = texts.replace('&#39;', "'")
            # remove symbol names
            texts = re.sub(r'(\#)(\S+)', r'hashtag_\2', texts)
            texts = re.sub(r'(\$)([A-Za-z]+)', r'cashtag_\2', texts)
            # remove usernames
            texts = re.sub(r'(\@)(\S+)', r'mention_\2', texts)
            # demojize
            texts = emoji.demojize(texts, delimiters=("", " "))
            return texts.strip()

        for c in cols:
            self.df[c] = self.df[c].astype("string")
            self.df[c] = self.df[c].apply(process)
            self.df[c] = self.df[c].astype("string")

    def handleMissingVal():
        '''
        處理missing value
        '''
        pass

    def dropData(self, dropCols=[], dropByPrefix=[]):
        '''
        處理drop資料、改變資料的type
        '''
        self.df.drop(columns=dropCols, inplace=True)
        for col in dropByPrefix:
            self.df.drop(
                self.df.filter(regex=col), axis=1, inplace=True
            )

    def modifyDataType(self, cols: list, dataType: str):
        for col in cols:
            self.df[col] = self.df[col].astype(dataType)

    def ReadLabelData(self):
        self.df = pd.read_csv(self.readDir + self.fileName, parse_dates=['Date'])

    def writeResults(self):
        '''
        寫處理過的資料到檔案
        '''
        # print(" Writing Results [V]")
        if not os.path.exists(self.resultDir):
            os.makedirs(self.resultDir)
        self.df.to_csv(self.resultDir + self.fileName)

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
# - Conversation
#   - replies
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
        "id,Date,user,likes,reshares,conversation,reshare_message".split(','))
    TweetLoader.writeResults()
    print(TweetLoader.result_df.head())
    print("======= End ========")
    print("Data shape: ", TweetLoader.result_df.shape)


def preprocessing():
    read_dir='./data/TSLA_2020_2022/labeled_data/'
    result_dir='./data/TSLA_2020_2022/processed_data/'
    fileNames = [
        f for f in os.listdir(read_dir)
        if f.endswith(".csv")
    ]
    # fileNames = fileNames[:1]
    for fileName in fileNames:
        pre = Preprocessing(
            read_dir=read_dir, result_dir=result_dir, fileName=fileName
        )
        pre.ReadLabelData()
        pre.modifyDataType(cols=["label", "sentences"], dataType="string")
        pre.process_text(["sentences"])
        pre.extendJsonCols()
        pre.dropData(
            dropCols=[
                "Unnamed: 0",
                "user.id",
                "user.username",
                "user.name",
                "user.avatar_url",
                "user.avatar_url_ssl",
                "user.identity",
                "user.classification",
                "user.plus_tier",
                "user.premium_room",
                "user.trade_app",
                "user.trade_status",
                "likes.user_ids",
                "conversation.parent_message_id",
                "conversation.in_reply_to_message_id",
                "conversation.parent",
                "reshares.user_ids",
                "reshare_message.reshared_deleted",
                "reshare_message.reshared_user_deleted",
                "reshare_message.parent_reshared_deleted",
            ],
            dropByPrefix=[
                "Unnamed: 0",
                "reshare_message.message",
                "user.portfolio"
            ],
        )

        pre.modifyDataType(cols=["user.join_date"], dataType="string")
        pre.df["user.join_date"] = pd.to_datetime(
            pre.df["user.join_date"], 
            format = "%Y-%m-%d", 
            errors = "coerce"
        ).astype("int64")

        pre.df["likes.total"] = pre.df["likes.total"].fillna(0)
        pre.df["conversation.replies"] = pre.df["conversation.replies"].fillna(0)
        pre.modifyDataType(
            cols=["likes.total", "conversation.replies"], dataType="int64"
        )

        pre.df["reshare"] = pre.df.apply(
            lambda x: x["reshares.reshared_count"] > 0 and x["reshare_message.reshared_count"] > 0, axis=1
        ).astype("bool")
        pre.dropData(dropCols=["reshares.reshared_count", "reshare_message.reshared_count"])
        pre.writeResults()
    pre.printInfo()

def create_features():
    featureloader = FeatureLoader(
        datasetDir="./data/TSLA_2020_2022/processed_data/",
        featuresDir="./data/TSLA_2020_2022/features_test/",
        n_arms=10,
        n_features=10
    )
    featureloader.loadData()
    featureloader.modifyDataType(
        cols=["user.official", "reshare"], dataType="int64"
    )
    featureloader.scaleData(
        standard_scale_columns=[
            "user.official",
            "user.followers",
            "user.following",
            "user.ideas",
            "user.watchlist_stocks_count",
            "user.like_count",
            "likes.total",
            "conversation.replies",
            "reshare",
        ],
        min_max_scale_columns=["user.join_date"],
        other_columns=["Date", "label", "score"],
    )
    featureloader.createFeatures()
    featureloader.writeFeatures()


if __name__ == '__main__':
    start_time = datetime.now()
    # tweet()
    # preprocessing()
    create_features()
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))
