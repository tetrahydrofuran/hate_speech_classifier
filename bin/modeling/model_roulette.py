import logging

import pandas as pd
from sklearn.externals import joblib

from . import *


def model_roulette(dataframe, label,
                   columns=('tweet', 'bigram', 'trigram', 'pos', 'bigram_pos', 'trigram_pos'),
                   size=0.3, state=42, binary=False):
    list_models = []
    reports = []
    for column in columns:
        logging.debug('model_roulette(): Processing column ' + column)
        for i in range(5):
            clf, test_feat, test_result = loop_models(i, dataframe, column, size, state)
            list_models.append(clf)
            report = custom_report.classification_report(test_result, clf.predict(test_feat))
            reports.append(report)

            logging.debug(clf.__class__.__name__ + ' ' + column)
            logging.debug(report)

    logging.debug("model_roulette(): Dumping reports into '../data/model_reports.pkl'")
    joblib.dump(reports, '../data/models/model_reports.pkl')
    reports = format_report(reports, label, binary)
    joblib.dump(reports, '../data/models/model_reports.pkl')
    return reports


def loop_models(index, dataframe, column, size, state):
    if index == 0:
        logging.debug('loop_models(): GaussianNB')
        return naive_bayes.train_naive_bayes(dataframe, column, size, state)
    if index == 1:
        logging.debug('loop_models(): DecisionTreeClassifier')
        return trees.train_decision_tree(dataframe, column, size, state)
    if index == 2:
        logging.debug('loop_models(): SGDClassifier')
        return sgd.train_sgd(dataframe, column, size, state)
    if index == 3:
        logging.debug('loop_models(): GradientBoostedClassifier')
        return gbc.train_gbc(dataframe, column, size, state)
    if index == 4:
        logging.debug('loop_models(): LogisticRegression')
        return logistic.train_logistic(dataframe, column, size, state)


def format_report(reports, label, binary=False):
    # region Initialization
    label = [label]
    models = ['Naive Bayes', 'Decision Tree', 'Stochastic Gradient Desc.', 'Gradient Boosted Trees',
              'Logistic Regression']
    types = ['tweet', 'bigram', 'trigram', 'pos', 'bigram_pos', 'trigram_pos']
    c0p = []
    c1p = []
    c2p = []
    c0r = []
    c1r = []
    c2r = []
    c0f = []
    c1f = []
    c2f = []
    avgp = []
    avgr = []
    avgf = []
    # endregion

    # region Loops
    for report in reports:
        for i in range(3):
            if i == 0:
                c0p.append(report[0][i])
                c0r.append(report[1][i])
                c0f.append(report[2][i])
                avgp.append(report[4][i])
            if i == 1:
                if binary:
                    c2p.append(report[0][i])
                    c2r.append(report[1][i])
                    c2f.append(report[2][i])
                    avgr.append(report[4][1])
                    avgf.append(report[4][2])
                    continue
                else:
                    c1p.append(report[0][i])
                    c1r.append(report[1][i])
                    c1f.append(report[2][i])
                    avgr.append(report[4][i])
            if i == 2:
                if binary:
                    continue
                c2p.append(report[0][i])
                c2r.append(report[1][i])
                c2f.append(report[2][i])
                avgf.append(report[4][i])
    # endregion

    df = pd.DataFrame(
        [label * 30, models * 6, types * 5, c0p, c0r, c0f, c1p, c1r, c1f, c2p, c2r, c2f, avgp, avgr, avgf]).T
    df.columns = ['version', 'model', 'param', 'c0p', 'c0r', 'c0f', 'c1p', 'c1r', 'c1f', 'c2p', 'c2r', 'c2f', 'avgp',
                  'avgr', 'avgf']
    return df
