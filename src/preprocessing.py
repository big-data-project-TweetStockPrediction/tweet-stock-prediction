from tweet_score.dataLoader import DataLoader
from tweet_score.model import TweetAnalyzer
from datetime import datetime
# Selected Column

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


def main():
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


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))