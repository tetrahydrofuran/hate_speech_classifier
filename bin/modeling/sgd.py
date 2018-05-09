import logging
import os

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from . import tfidf


def train_sgd(df, column, size=0.3, state=42):
    filepath = '../data/models/sgd_' + column + '.pkl'
    Xtr, Xte, Ytr, Yte = tfidf.generate_tfidf_split(df, column, size, state)

    scaler = StandardScaler(with_mean=False)
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    params = {'loss': ['hinge', 'modified_huber', 'perceptron'], 'penalty': ['none', 'l2', 'l1'],
              'alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

    clf = GridSearchCV(SGDClassifier(max_iter=5), param_grid=params)

    logging.debug('train_sgd(): Training model')
    clf.fit(Xtr, Ytr)
    logging.debug('train_sgd(): Best parameters for column : ' + column)
    logging.debug(clf.best_params_)

    logging.debug("train_sgd(): Dumping SGDClassifier to '" + filepath + "'")
    joblib.dump(clf.best_estimator_, filepath)
    return clf.best_estimator_, Xte, Yte


def get_sgd(suffix):
    filepath = '../data/models/sgd_' + suffix + '.pkl'
    if os.path.isfile(filepath):
        logging.debug('get_sgd(): Loading saved classifier.')
        return joblib.load(filepath)
    else:
        logging.error('get_sgd(): Saved classifier not found.')
        return None
