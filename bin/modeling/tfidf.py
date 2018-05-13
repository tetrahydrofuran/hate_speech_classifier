import logging

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from processing.normalize import rejoin

doPCA = True
pcaSize = 50


def pcaSetter(pca, size):
    """Sets global PCA variable to determine whether or not to conduct PCA"""
    global doPCA
    global pcaSize
    doPCA = pca
    pcaSize = size


def generate_tfidf_split(df, column='tweet', size=0.3, state=42):
    """
    If I could do this over again, I would return the TfidfVectorizer object, and not the train-test split.
    Applies tf-idf weights to vocabulary and returns a train-test split.
    :param df: pandas DataFrame
    :param column: Column to process
    :param size: Size of test set
    :param state: Random state for train-test split
    :return: 4 variables representing a train test split
    """
    logging.debug('generate_tfidf_split(): Transforming column: ' + column)
    train, test = train_test_split(df, test_size=size, random_state=state, stratify=df['class'])
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train[column].apply(rejoin))
    Y_train = train['class']
    X_test = vectorizer.transform(test[column].apply(rejoin))
    Y_test = test['class']

    if doPCA:
        logging.debug('generate_tfidf_split(): Applying PCA transformer')
        pca = PCA(n_components=pcaSize)
        X_train = pca.fit_transform(X_train.toarray())
        X_test = pca.transform(X_test.toarray())

    return X_train, X_test, Y_train, Y_test
