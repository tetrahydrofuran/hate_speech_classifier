from sklearn.tree import tree
from sklearn.externals import joblib
import logging
import os
from . import tfidf


def train_decision_tree(df, column, size=0.3, state=42):
    clf = tree.DecisionTreeClassifier()
    filepath = '../data/models/decision_tree_' + column + '.pkl'
    Xtr, Xte, Ytr, Yte = tfidf.generate_tfidf_split(df, column, size, state)

    logging.debug('train_decision_tree(): Training model')
    clf.fit(Xtr.toarray(), Ytr)
    logging.debug("train_decision_tree(): Dumping DecisionTreeClassifier to '" + filepath + "'")
    joblib.dump(clf, filepath)
    return clf, Xte, Yte


def get_decision_tree(suffix):
    filepath = '../data/models/decision_tree_' + suffix + '.pkl'
    if os.path.isfile(filepath):
        logging.debug('get_decision_tree(): Loading saved classifier.')
        return joblib.load(filepath)
    else:
        logging.error('get_decision_tree(): Saved classifier not found.')
        return None
