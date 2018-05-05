import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import logging
import os


def train_classifier():
    logging.debug('Entering train_classifier()')
    tagged_sentences = brown.tagged_sents()
    feature_set = []
    logging.debug('train_classifier(): Extracting features from corpus')
    for sentence in tagged_sentences:
        untagged = nltk.tag.untag(sentence)
        for index, (word, tag) in enumerate(sentence):
            feature_set.append((word_feature_extraction(untagged, index), tag))

    train, test = train_test_split(feature_set, test_size=0.3, random_state=42)
    classifier = nltk.NaiveBayesClassifier.train(train)
    logging.debug("train_classifier(): Dumping trained NB to '../data/models/pos_classifier.pkl'")
    joblib.dump(classifier, '../data/models/pos_classifier.pkl')

    # For some reason, doing nltk.classify.accuracy hangs up the process and prevents completion;
    # added random state to enable validation separately
    # logging.debug("train_classifier(): Checking accuracy against test set")
    # print(nltk.classify.accuracy(classifier, test))




def word_feature_extraction(sentence, index):
    word = sentence[index]
    previous_word = "<START>"
    next_word = "<END>"
    if index > 0:
        previous_word = sentence[index - 1]
    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
    suffix1 = word[-1]
    suffix2 = word[-2:]
    suffix3 = word[-3:]
    numeric = True if word.isnumeric() else False

    feat_dict = {
        "previous": previous_word,
        "next": next_word,
        "suffix1": suffix1,
        "suffix2": suffix2,
        "suffix3": suffix3,
        "numeric": numeric
    }
    return feat_dict
