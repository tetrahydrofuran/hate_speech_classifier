import logging

from sklearn.externals import joblib

from . import *


def model_roulette(dataframe,
                   columns=('tweet', 'bigram', 'trigram', 'pos', 'bigram_pos', 'trigram_pos'),
                   size=0.3, state=42):
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
    joblib.dump(reports, '../data/model_reports.pkl')
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