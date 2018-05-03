import logging
import pickle

from nltk.corpus import treebank
from sklearn import tree
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def load_pos_classifier(name='bayes'):
    return pickle.load(open('../data/pos_' + name + '_classifier.pkl', 'rb'))


# Due to memory constraints, data available to the model is limited to 1500 sentences
def export_pos_classifier(classifier, state=42):
    logging.debug('Entering export_pos_classifier()')
    tagged = treebank.tagged_sents()[:1500]
    X_features = []
    for sentence in tagged:
        for k in range(len(sentence)):
            X_features.append(word_feature_extraction(sentence, k))
    vectorizer = DictVectorizer(sparse=False)
    logging.debug('exporting_pos_classifier(): Transforming features')
    X = vectorizer.fit_transform(X_features)
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


# Certain features were extracted, but removed due to memory constraints
def word_feature_extraction(tagged_sentence, index):
    word, tag = tagged_sentence[index]
    lower = word.lower()
    previous_word = ""
    next_word = ""
    if index > 0:
        previous_word = tagged_sentence[index - 1][0]
    if index < len(tagged_sentence) - 1:
        next_word = tagged_sentence[index + 1][0]
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
