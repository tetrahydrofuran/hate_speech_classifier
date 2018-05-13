import logging
import os

from sklearn.externals import joblib
from sklearn.tree import tree

from . import tfidf


def train_decision_tree(df, column, size=0.3, state=42):
    """Creates and trains a decision tree classifier using tf-idf weights"""
    clf = tree.DecisionTreeClassifier()
    filepath = '../data/models/decision_tree_' + column + '.pkl'
    Xtr, Xte, Ytr, Yte = tfidf.generate_tfidf_split(df, column, size, state)

    logging.debug('train_decision_tree(): Training model')
    clf.fit(Xtr, Ytr)
    logging.debug("train_decision_tree(): Dumping DecisionTreeClassifier to '" + filepath + "'")
    joblib.dump(clf, filepath)
    return clf, Xte, Yte


def get_decision_tree(suffix):
    """Loads and returns decision tree trained on the specified column"""
    filepath = '../data/models/decision_tree_' + suffix + '.pkl'
    if os.path.isfile(filepath):
        logging.debug('get_decision_tree(): Loading saved classifier.')
        return joblib.load(filepath)
    else:
        logging.error('get_decision_tree(): Saved classifier not found.')
        return None
