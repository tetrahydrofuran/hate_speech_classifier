import logging
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

from . import tfidf


def train_gbc(df, column, size=0.3, state=42):
    """Creates and trains a gradient boosted classifier using tf-idf weights"""
    filepath = '../data/models/gbc_' + column + '.pkl'
    Xtr, Xte, Ytr, Yte = tfidf.generate_tfidf_split(df, column, size, state)

    clf = GradientBoostingClassifier(max_depth=10)

    logging.debug('train_gbc(): Training model')
    clf.fit(Xtr, Ytr)

    logging.debug("train_gbc(): Dumping GradientBoostingClassifier to '" + filepath + "'")
    joblib.dump(clf, filepath)
    return clf, Xte, Yte


def get_gbc(suffix):
    """Loads and returns gradient boosted classifier trained on the specified column"""
    filepath = '../data/models/gbc_' + suffix + '.pkl'
    if os.path.isfile(filepath):
        logging.debug('get_gbc(): Loading saved classifier.')
        return joblib.load(filepath)
    else:
        logging.error('get_gbc(): Saved classifier not found.')
        return None
