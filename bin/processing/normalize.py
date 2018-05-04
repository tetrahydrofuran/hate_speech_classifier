import logging
import os
import re
import string

import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from . import spelling as sp


def process_tweets(df_tweets, colname='tweet', reprocess=False):
    """
    Processes, stems, parses, and normalizes tweet text, including separating emojis and hashtags
    :param df_tweets: pandas DataFrame
    :param colname: String name of column containing tweet
    :param reprocess: Boolean to redo processing steps if postprocessed frame does not exist
    :return: TODO fill out
    """
    logging.debug('Entering process_tweet()')
    if not os.path.isfile('../data/postprocessed.pkl') or reprocess:
        # Cleaning
        df_tweets[colname] = df_tweets[colname].apply(case_correction)
        df_tweets[colname] = df_tweets[colname].apply(remove_mentions)
        df_tweets[colname] = df_tweets[colname].apply(remove_retweets)
        df_tweets['emoji'] = df_tweets[colname].apply(emoji_extraction)
        df_tweets[colname] = df_tweets[colname].apply(emoji_removal)
        df_tweets['hashtag'] = df_tweets[colname].apply(hashtag_extraction)
        df_tweets[colname] = df_tweets[colname].apply(hashtag_removal)
        df_tweets[colname] = df_tweets[colname].apply(clean_special_characters)
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

        df_tweets[colname] = df_tweets[colname].apply(porter_stemming)
        # TODO insert shallow parse, if time
        df_tweets[colname] = df_tweets[colname].apply(stopword_removal)
        df_tweets['bigram'] = df_tweets[colname].apply(bigram_creation)
        df_tweets['trigram'] = df_tweets[colname].apply(trigram_creation)
        logging.debug("Dumping processed dataframe into '../data/postprocessed.pkl'")
        joblib.dump(df_tweets, '../data/postprocessed.pkl')
    else:
        logging.debug("process_tweet(): Bypassing processing step, loading 'postprocessed.pkl'.")
        df_tweets = joblib.load('../data/postprocessed.pkl')
    # region Generate or Load Class-Separated Features
    if not (os.path.isfile('../data/corpus/class_0_tweet_list.pkl') and
            os.path.isfile('../data/corpus/class_1_tweet_list.pkl') and
            os.path.isfile('../data/corpus/class_2_tweet_list.pkl')):
        class0_words, class1_words, class2_words = generate_corpus(df_tweets, col='tweet')
        class0_emoji, class1_emoji, class2_emoji = generate_corpus(df_tweets, col='emoji')
        class0_hash, class1_hash, class2_hash = generate_corpus(df_tweets, col='hashtag')
    else:
        class0_words = joblib.load('../data/corpus/class_0_tweet_list.pkl')
        class1_words = joblib.load('../data/corpus/class_1_tweet_list.pkl')
        class2_words = joblib.load('../data/corpus/class_2_tweet_list.pkl')
        class0_emoji = joblib.load('../data/corpus/class_0_emoji_list.pkl')
        class1_emoji = joblib.load('../data/corpus/class_1_emoji_list.pkl')
        class2_emoji = joblib.load('../data/corpus/class_2_emoji_list.pkl')
        class0_hash = joblib.load('../data/corpus/class_0_hashtag_list.pkl')
        class1_hash = joblib.load('../data/corpus/class_1_hashtag_list.pkl')
        class2_hash = joblib.load('../data/corpus/class_2_hashtag_list.pkl')
    # endregion


# region Helper Functions
# region Pre-tokenizing Workflow
def remove_mentions(element):
    """Remove twitter mentions '@name' from text"""
    return re.sub(r'@[a-zA-Z_0-9]{1,15}', '', element)


def remove_retweets(element):
    """Removes 'RT:', representing retweets"""
    return re.sub(r'RT ?:?', '', element, flags=re.IGNORECASE)


# region Feature extraction and removal
def emoji_extraction(text):
    """Returns unicode emojis"""
    return re.findall(r'&#\d+;?', text)


def emoji_removal(text):
    """Removes unicode emojis from string"""
    return re.sub(r'&#\d+;?', '', text)


def hashtag_extraction(text):
    """Returns twitter hashtags"""
    return re.findall(r'#\w+', text)


def hashtag_removal(text):
    """Removes twitter hashtags from string"""
    return re.sub(r'#\w+', '', text)
# endregion


def clean_special_characters(text):
    """Substitutes 'and' for ampersands, removes hyperlinks and punctuation"""
    text = re.sub(r'&amp', 'and', text)
    text = re.sub(r'http[a-zA-Z0-9:/.-]+', '', text)  # Remove hyperlinks
    # Remove punctuation
    punct = re.compile('[{}]'.format(re.escape(string.punctuation)))
    text = re.sub(punct, '', text)
    return text
# endregion


# region Tokenizing, n-gram Creation
def tokenize(corpus):  # or just word tokenizing
    """Tokenizes / creates unigrams from text"""
    return nltk.WhitespaceTokenizer().tokenize(corpus)


def bigram_creation(corpus):
    """Creates list of bigrams from text"""
    return list(zip(corpus, corpus[1:]))


def trigram_creation(corpus):
    """Creates list of trigrams from text"""
    return list(zip(corpus, corpus[1:], corpus[2:]))
# endregion


# region Post-tokenizing Workflow
def generate_corpus(df, col):
    """
    Gathers words present in each class of tweets, dumping to pkl and returning list if exists
    :param df: Passed in tweet dataframe
    :return: List of list of words
    """
    logging.debug('Entering generate_corpus() for column ' + col + '.')
    output = []
    for i in range(3):
        logging.debug('generate_corpus(): Iteration ' + str(i + 1) + ' of 3.')
        x = df[df['class'] == i][col]
        words = []
        for item in x:
            for word in item:
                words.append(word)
        joblib.dump(words, '../data/corpus/class_' + str(i) + '_' + col + 'list.pkl')
        output.append(words)
    logging.debug('Exiting generate_corpus()')
    return output[0], output[1], output[2]


def case_correction(text):
    """Returns lowercase string"""
    return text.lower()


def stopword_removal(text):
    """Removes stopwords from list of words as determined by nltk"""
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in text if word not in stopwords]


def porter_stemming(text):
    """Applies Porter stemming algorithm"""
    pstem = PorterStemmer()
    return [pstem.stem(word) for word in text]

# endregion
# endregion
