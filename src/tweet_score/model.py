from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline
import pandas as pd
from dask import dataframe as dd
import emoji
import re
import pprint
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
            seq.append(row["body"])
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
        # sentences = pd.Series(sentences)
        # if input text contains https, @ or # or $ symbols, better apply preprocess to get a more accurate result
        for idx, seq in self.batchSentences(df, batch_size):
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
    tweets = ['#just buy', 'just sell it',
              'entity rocket to the sky!',
              'go down', 'even though it is going up, I still think it will not keep this trend in the near future']
    res = Analyzer.tokenize(tweets, needProcessed=True)
    pprint.pprint(res)
