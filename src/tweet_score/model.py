from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline
import pandas as pd
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

    def process_text(self, texts):
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

    def tokenize(self, sentences: list, needProcessed=False):
        sentences = pd.Series(sentences)
        # if input text contains https, @ or # or $ symbols, better apply preprocess to get a more accurate result
        sentences = list(sentences.apply(self.process_text)
                         ) if needProcessed else list(sentences)
        results = self.nlp(sentences)
        return results  # 2 labels, label 0 is bearish, label 1 is bullish


if __name__ == '__main__':
    Analyzer = TweetAnalyzer()
    tweets = ['#just buy', 'just sell it',
              'entity rocket to the sky!',
              'go down', 'even though it is going up, I still think it will not keep this trend in the near future']
    res = Analyzer.tokenize(tweets, needProcessed=True)
    pprint.pprint(res)
