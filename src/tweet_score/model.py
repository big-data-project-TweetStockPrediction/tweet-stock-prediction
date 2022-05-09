from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline
import pandas as pd
from dask import dataframe as dd
import emoji
import re
import pprint
from tqdm import tqdm
# the model was trained upon below preprocessing


class TweetAnalyzer(object):
    def __init__(self, modelName='zhayunduo/roberta-base-stocktwits-finetuned'):
        super(TweetAnalyzer, self).__init__()
        tokenizer_loaded = RobertaTokenizer.from_pretrained(modelName)
        model_loaded = RobertaForSequenceClassification.from_pretrained(
            modelName)
        self.nlp = pipeline("text-classification", model=model_loaded,
                            tokenizer=tokenizer_loaded)

    def process_text(self, texts: str):
        # remove URLs
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
        # print(texts.strip())
        return texts.strip()

    def batchSentences(self, sentences: pd.DataFrame, batch_size=512):
        seq = []
        idx = []

        for _, row in sentences.iterrows():
            seq.append(' '.join(row["body"].split(' ')[:400]))
            idx.append(row["id"])
            if len(seq) == batch_size:
                yield idx, seq
                seq = []
                idx = []
        yield idx, seq

    def batchTokenize(self, df: pd.DataFrame, batch_size=512, needProcessed=False):
        """fit sentences to model for predicting bearish(label 0) or bullish(label 1).

        Args:
            sentences (list): sentences we want to predict.
            needProcessed (bool, optional): If you want process some meaningless symbol or link in sentence, you should set True. Defaults to False.

        Returns:
            list: labeled results with sentences
        """
        with tqdm(total=int(len(df)/batch_size)) as pbar:
            for idx, seq in self.batchSentences(df, batch_size):
                pbar.update(1)
                series = pd.Series(seq)
                sents = list(
                    series.apply(self.process_text)
                ) if needProcessed else list(sents)
                results = self.nlp(sents)
                scoreDict = {
                    k: [dic[k] for dic in results]
                    for k in results[0]
                }
                scoreDict["sentences"] = seq
                scoreDict["id"] = idx
                yield dd.from_pandas(pd.DataFrame(data=scoreDict), chunksize=512).set_index("id")


# if __name__ == '__main__':
def main():
    Analyzer = TweetAnalyzer()
    tweets = ['even though it is going up, I still think it will not keep this trend in the near future even though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near futureeven though it is going up, I still think it will not keep this trend in the near future']
    for idx, t in enumerate(tweets):
        tweets[idx] = ' '.join(t.split(" ")[:400])
    results = Analyzer.nlp(tweets)
    pprint.pprint(results)


if __name__ == '__main__':
    main()
