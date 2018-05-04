import logging
import os
import re
import string
from collections import Counter

import nltk

from . import spelling as sp


def process_tweet(df_tweets, colname = 'tweet'):
    """
    TODO: Fill out
    :param df_tweets: pandas DataFrame
    :param colname: String name of column containing tweet
    :return:
    """
    logging.debug('Entering process_tweet()')

    # Cleaning
    df_tweets[colname] = df_tweets[colname].apply(remove_mentions)
    df_tweets[colname] = df_tweets[colname].apply(remove_retweets)
    df_tweets['emoji'] = df_tweets[colname].apply(emoji_extraction)
    df_tweets[colname] = df_tweets[colname].apply(emoji_removal)
    df_tweets[colname] = df_tweets[colname].apply(clean_special_characters)
    df_tweets['hashtag'] = df_tweets[colname].apply(hashtag_extraction)
    df_tweets[colname] = df_tweets[colname].apply(hashtag_removal)

    # Tokenizing
    df_tweets[colname] = df_tweets[colname].apply(tokenize)
    # region Spelling workflow
    logging.debug('process_tweet(): Entering spelling workflow')
    dict_location = '../data/wordlist.pkl'
    if os.path.isfile(dict_location):
        word_dict = sp.load_dictionary(dict_location)
    else:
        sp.generate_dictionary(dict_location)
        word_dict = sp.load_dictionary(dict_location)
    df_tweets[colname] = df_tweets[colname].apply(sp.remove_repeated_characters)
    df_tweets[colname] = df_tweets[colname].apply(sp.spelling_normalization, args=(word_dict, ))

    # endregion
    df_tweets['bigram'] = df_tweets[colname].apply(bigram_creation)
    df_tweets['trigram'] = df_tweets[colname].apply(trigram_creation)

# region Pre-tokenizing Workflow
def remove_mentions(element):
    return re.sub(r'@[a-zA-Z_0-9]{1,15}', '', element)


def remove_retweets(element):
    return re.sub(r'RT ?:?', '', element)


# region Feature extraction and removal
def emoji_extraction(text):
    return re.findall(r'&#\d+;?', text)


def emoji_removal(text):
    return re.sub(r'&#\d+;?', '', text)


def hashtag_extraction(text):
    return re.findall(r'#\w+', text)


def hashtag_removal(text):
    return re.sub(r'#\w+', '', text)
# endregion


def clean_special_characters(text):
    text = re.sub(r'&amp', 'and', text)
    text = re.sub(r'http[a-zA-Z0-9:/.-]+', '', text)  # Remove hyperlinks
    # Remove punctuation
    punct = re.compile('[{}]'.format(re.escape(string.punctuation)))
    text = re.sub(punct, '', text)
    return text
# endregion


# region Tokenizing, n-gram creation
def tokenize(corpus):  # or just word tokenizing
    return nltk.WhitespaceTokenizer().tokenize(corpus)


def bigram_creation(corpus):
    return list(zip(corpus, corpus[1:]))


def trigram_creation(corpus):
    return list(zip(corpus, corpus[1:], corpus[2:]))
# endregion


# region Post-tokenizing Workflow
def generate_corpus(text):
    # TODO: check if necessary or if incorporated in tf-idf workflow
    """
    Creates list of words, as well as a Counter object of each word's count.
    :param text: Assumes passing in a pandas Series comprised of tokenized list of words in each entry
    :return: words: list of words, counts: Counter object of word frequency
    """
    words = []
    for entry in text:
        for word in entry:
            words.append(word)
    counts = Counter(words)
    return words, counts


def case_correction(text):
    return text.lower()


def stopword_removal(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in text if word not in stopwords]

# endregion
