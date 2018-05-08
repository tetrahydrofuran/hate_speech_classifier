from sklearn.externals import joblib
import logging
import os

# This was refactored out of normalize.py, and so some things are kind of unnecessary artifacts of that
# but were too much effort to change

# TODO: Refactor to be relevant now
def process_classes(df, column):
    # region Generate or Load Class-Separated Features
    if not (os.path.isfile('../data/corpus/class_0_tweet_list.pkl') and
            os.path.isfile('../data/corpus/class_1_tweet_list.pkl') and
            os.path.isfile('../data/corpus/class_2_tweet_list.pkl')):
        generate_corpus(df, col='tweet')
        generate_corpus(df, col='emoji')
        generate_corpus(df, col='hashtag')
    if column == 'tweet':
        class0_words = joblib.load('../data/corpus/class_0_tweet_list.pkl')
        class1_words = joblib.load('../data/corpus/class_1_tweet_list.pkl')
        class2_words = joblib.load('../data/corpus/class_2_tweet_list.pkl')
        return class0_words, class1_words, class2_words
    if column == 'emoji':
        class0_emoji = joblib.load('../data/corpus/class_0_emoji_list.pkl')
        class1_emoji = joblib.load('../data/corpus/class_1_emoji_list.pkl')
        class2_emoji = joblib.load('../data/corpus/class_2_emoji_list.pkl')
        return class0_emoji, class1_emoji, class2_emoji
    if column == 'hashtag':
        class0_hash = joblib.load('../data/corpus/class_0_hashtag_list.pkl')
        class1_hash = joblib.load('../data/corpus/class_1_hashtag_list.pkl')
        class2_hash = joblib.load('../data/corpus/class_2_hashtag_list.pkl')
        return class0_hash, class1_hash, class2_hash
    # endregion


def generate_corpus(df, col):
    """
    Gathers words present in each class of tweets, dumping to pkl and returning list if exists
    :param df: Passed in tweet dataframe
    :return: None
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

