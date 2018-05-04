import logging
import os

from nltk.corpus import treebank
from sklearn import tree
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

"""
These classifiers were trained too specifically, and are not generalizable to new information.
A new classifier will be constructed, and features more carefully selected.
"""

def tag_pos(data, model):
    if model:
        if not os.path.isfile('../data/pos_bayes_classifier.pkl'):
            export_pos_classifier(classifier='bayes', state=42)
        clf = load_pos_classifier('bayes')
    else:
        if not os.path.isfile('../data/pos_tree_classifier.pkl'):
            export_pos_classifier(classifier='tree', state=42)
        clf = load_pos_classifier('tree')
    # TODO do the classifying


def load_pos_classifier(name):
    return joblib.load('../data/pos_' + name + '_classifier.pkl')


# Due to memory constraints, data available to the model is limited to 1500 sentences
def export_pos_classifier(classifier, state=42):
    logging.debug('Entering export_pos_classifier()')
    tagged = treebank.tagged_sents()[:1500]
    X = pos_transformer(tagged)
    Y = []
    logging.debug('exporting_pos_classifier(): Transforming response')
    for sentence in tagged:
        for k in range(len(sentence)):
            Y.append(sentence[k][1])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=state)
    logging.debug("exporting_pos_classifier(): Dumping test set to 'pos_testX.pkl', 'pos_testY.pkl'")
    joblib.dump(X_test, '../data/pos_testX.pkl')
    joblib.dump(Y_test, '../data/pos_testY.pkl')
    if classifier == 'tree':
        logging.debug('exporting_pos_classifier(): Fitting tree model')
        tree_clf = tree.DecisionTreeClassifier(criterion='entropy')
        tree_clf.fit(X_train, Y_train)

        logging.debug("exporting_pos_classifier(): Dumping tree model to 'pos_tree_classifier.pkl'")
        joblib.dump(tree_clf, '../data/pos_tree_classifier.pkl')
    if classifier == 'bayes':
        logging.debug('exporting_pos_classifier(): Fitting naive Bayes model')
        bayes_clf = GaussianNB()
        bayes_clf.fit(X_train, Y_train)

        logging.debug("exporting_pos_classifier(): Dumping naive Bayes model to 'pos_bayes_classifier.pkl'")
        joblib.dump(bayes_clf, '../data/pos_bayes_classifier.pkl')


def pos_transformer(data):
    logging.debug('pos_transformer(): Transforming features')
    X_features = []
    for sentence in data:
        for k in range(len(sentence)):
            X_features.append(word_feature_extraction(sentence, k))
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(X_features)

    # TODO refactor X


# Certain features were extracted, but removed due to memory constraints
def word_feature_extraction(sentence, index):
    """
    Used in both training and on unseen data; try block is for training data, except for new data
    # TODO fill in
    :param sentence:
    :param index:
    :return:
    """
    try:
        word, tag = sentence[index]
    except ValueError:
        word = sentence[index]
    lower = word.lower()
    previous_word = ""
    next_word = ""
    if index > 0:
        try:
            previous_word = sentence[index - 1][0]
        except ValueError:
            previous_word = sentence[index - 1]
    if index < len(sentence) - 1:
        try:
            next_word = sentence[index + 1][0]
        except:
            next_word = sentence[index + 1]
    suffix1 = word[-1]
    # suffix2 = word[-2:]
    suffix3 = word[-3:]
    # capitals = True if word.upper() == word else False
    numeric = True if word.isnumeric() else False

    feat_dict = {
        # "word": word,
        "lower": lower,
        "previous": previous_word,
        "next": next_word,
        "suffix1": suffix1,
        # "suffix2": suffix2,
        "suffix3": suffix3,
        # "capital": capitals,
        "numeric": numeric
    }
    return feat_dict
