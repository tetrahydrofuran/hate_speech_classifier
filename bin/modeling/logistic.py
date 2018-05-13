import logging
import os

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from . import tfidf


def train_logistic(df, column, size=0.3, state=42):
    """Creates and trains a gradient boosted classifier using tf-idf weights"""
    filepath = '../data/models/logistic_' + column + '.pkl'
    Xtr, Xte, Ytr, Yte = tfidf.generate_tfidf_split(df, column, size, state)

    params = {'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3], 'multi_class': ['ovr', 'multinomial'],
              'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']}
    clf = GridSearchCV(LogisticRegression(), param_grid=params)
    logging.debug('train_logistic(): Training model')
    clf.fit(Xtr, Ytr)

    logging.debug('train_logistic(): Best parameters for column : ' + column)
    logging.debug(clf.best_params_)

    logging.debug("train_logistic(): Dumping LogisticRegression to '" + filepath + "'")
    joblib.dump(clf.best_estimator_, filepath)
    return clf, Xte, Yte


def get_logistic(suffix):
    """Loads and returns gradient boosted classifier trained on the specified column"""
    filepath = '../data/models/logistic_' + suffix + '.pkl'
    if os.path.isfile(filepath):
        logging.debug('get_logistic(): Loading saved classifier.')
        return joblib.load(filepath)
    else:
        logging.error('get_logistic(): Saved classifier not found.')
        return None
