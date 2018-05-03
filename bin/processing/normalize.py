import logging
import re
import string
from collections import Counter

import nltk
from nltk.corpus import wordnet


def process_text(incoming):
    # TODO: Logging
    logging.debug('Incoming text: ')
    logging.debug('')
    pass
    logging.debug('Outgoing text: ')
    logging.debug('')


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


def repeating_letter_removal(text):
    pass


def remove_repeated_characters(words):
    def get_real_word(word):
        if wordnet.synsets(word):
            return word
        new_word = repeats.sub(match_sub, word)
        return get_real_word(new_word) if new_word != word else new_word
    repeats = re.compile(r'(\w*)(\w)\2(\w*)')
    match_sub = r'\1\2\3'
    return [get_real_word(word) for word in words]


def spelling_normalization(text):
    pass
# endregion
