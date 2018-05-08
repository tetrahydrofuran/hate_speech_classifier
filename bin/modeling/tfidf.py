from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from processing.normalize import rejoin
import logging


def generate_tfidf_split(df, column='tweet', size=0.3, state=42):
    logging.debug('generate_tfidf_split(): Transforming column: ' + column)
    train, test = train_test_split(df, test_size=size, random_state=state)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train[column].apply(rejoin))
    Y_train = train['class']
    X_test = vectorizer.transform(test[column].apply(rejoin))
    Y_test = test['class']
    return X_train, X_test, Y_train, Y_test


def rejoin(text):
    return ' '.join(text)
# tweet for bag-of-words


