import logging

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from processing.normalize import rejoin

doPCA = True
pcaSize = 50


def pcaSetter(pca, size):
    global doPCA
    global pcaSize
    doPCA = pca
    pcaSize = size


def generate_tfidf_split(df, column='tweet', size=0.3, state=42):
    """
    If I could do this over again, I would return the TfidfVectorizer object, and not the train-test split.
    :param df:
    :param column:
    :param size:
    :param state:
    :return:
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


def rejoin(text):
    return ' '.join(text)
