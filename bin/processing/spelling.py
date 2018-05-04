import logging
import re
from collections import Counter

from nltk.corpus import wordnet
from sklearn.externals import joblib


def generate_dictionary(location):
    """
    Creates dictionary from a very, very, very large text corpus and dumps to a pkl.
    :return: None
    """
    f = open('../data/wordlist.txt', 'rb')
    words = Counter(re.findall('[a-z]+', f.read().lower().decode()))
    joblib.dump(words, location)


def load_dictionary(location='../data/wordlist.pkl'):
    """
    Loads generated dictionary file, adds custom words
    :return: Counter object with word counts representing dictionary of words
    """
    words = joblib.load(location)
    # Simple plurals not necessary since removed in word correction
    manual_addition = ['sis', 'vixen' ,'ho', 'hoe', 'aint', 'nigga', 'nigger', 'bitch', 'bitches', 'lmao', 'pussy',
                       'fuck', 'fucking', 'fag', 'faggot', 'hater', 'lol', 'yall', 'nfl', 'fucked', 'nips', 'crap',
                       'whitey', 'ghetto', 'mac', 'tho', 'jamming', 'titties', 'dyke', 'fb', 'lmfao', 'bae', 'kanye',
                       'coon', 'smh', 'tweet', 'retweet', 'nbc', 'jk', 'im']
    for word in manual_addition:
        words[word] = 99999  # add word to counter
    return words


def remove_repeated_characters(word):
    """
    Removes repeated characters, terminating when no repeating characters to be found, or when word matches
    an existing word in the wordnet corpus.  Example: finallllyyyy -> finally
    :param word: Word to be processed
    :return: Actual word, or word without repeating character
    """
    def get_real_word(word):
        if wordnet.synsets(word):
            return word
        new_word = repeats.sub(match_sub, word)
        return get_real_word(new_word) if new_word != word else new_word
    repeats = re.compile(r'(\w*)(\w)\2(\w*)')
    match_sub = r'\1\2\3'
    return [get_real_word(word) for word in word]


def spelling_normalization(words, dictionary):
    """
    Replaces incorrectly-spelled word with most likely candidate based upon word frequency in dictionary corpus
    Code mostly taken from Text Analytics With Python (Apress)
    :param words: Word to be processed
    :return: Spell-corrected word
    """

    logging.debug('spelling_normalization(): Incoming words: ')
    logging.debug(words)

    def edits1(word):
        """Return 1-edits away"""
        letters = 'abcdefghijklmnopqrstuvwxyz'

        def split_word(word):
            """Returns all possible variations of word split up"""
            return [(word[:i], word[i:]) for i in range(len(word) + 1)]

        pairs = split_word(word)
        deletes = [a + b[1:]                      for (a, b) in pairs if b]
        transposes = [a + b[1] + b[0] + b[2:]     for (a, b) in pairs if len(b) > 1]
        replaces = [a + c + b[1:]                 for (a, b) in pairs for c in letters if b]
        inserts = [a + c + b                      for (a, b) in pairs for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word):
        """Return 2-edits away"""
        return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

    def known(words):
        return {w for w in words if w in dictionary}

    corrected = []
    for word in words:
        if word.isnumeric():
            corrected.append(word)
            continue
        if word in dictionary or wordnet.synsets(word):
            corrected.append(word)
            continue
        candidates = (known(edits1(word)) or
                      known(edits2(word)) or
                      [word])

        corrected.append(max(candidates, key=dictionary.get))
    logging.debug('spelling_normalization(): Outgoing words:')
    logging.debug(corrected)
    return corrected
