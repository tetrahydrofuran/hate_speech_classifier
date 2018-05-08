from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from . import tfidf

import logging
import os


def train_naive_bayes(df, column, size=0.3, state=42):
    clf = GaussianNB()
    filepath = '../data/models/naive_bayes_' + column + '.pkl'
    Xtr, Xte, Ytr, Yte = tfidf.generate_tfidf_split(df, column, size, state)

    logging.debug('train_naive_bayes(): Training model')
    clf.fit(Xtr.toarray(), Ytr)
    logging.debug("train_naive_bayes(): Dumping GaussianNB to '" + filepath + "'")
    joblib.dump(clf, filepath)
    return clf, Xte, Yte


def get_naive_bayes(suffix):
    filepath = '../data/models/naive_bayes_' + suffix + '.pkl'
    if os.path.isfile(filepath):
        logging.debug('get_naive_bayes(): Loading saved classifier.')
        return joblib.load(filepath)
    else:
        logging.error('get_naive_bayes(): Saved classifier not found.')
        return None
