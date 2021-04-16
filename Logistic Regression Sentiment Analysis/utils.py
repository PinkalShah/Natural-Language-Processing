import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def tweet_processing(tweet):
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')

    # remove tickers startswith $
    tweet = re.sub(r'\$\w','', tweet)
    # remove RT
    tweet = re.sub(r'^RT[\s]+','',tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet)
    #remove hashtags
    tweet = re.sub(r'#','',tweet)
    # tokenization
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    clean_tweet = []
    for word in tweet_tokens:
        if(word not in stopwords_en and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            clean_tweet.append(stem_word)

    return clean_tweet


def frequency_builder(tweets, ys):
    yslist = np.squeeze(ys).tolist()

    frequencies = {}
    for y, tweet in zip(yslist, tweets):
        for word in tweet_processing(tweet):
            pair = (word, y)
            if pair in frequencies:
                frequencies[pair] += 1
            else:
                frequencies[pair] = 1

    return frequencies