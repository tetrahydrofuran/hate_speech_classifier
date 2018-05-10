import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .tfidf import rejoin


# TODO Implement PCA


def generate_compound_model(df, size, state):
    models = [DecisionTreeClassifier(), DecisionTreeClassifier(), DecisionTreeClassifier(), GradientBoostingClassifier(
        max_depth=10), GradientBoostingClassifier(max_depth=10)]
    columns = ['tweet', 'bigram', 'trigram_pos', 'pos', 'trigram']

    train, test = train_test_split(df, test_size=size, random_state=state, stratify=df['class'])

    train_sets = []
    for column in columns:
        Xtr, Ytr, vectorizer, scaler = compound_tfidf(train, column)
        train_sets.append([Xtr, Ytr, vectorizer, scaler])

    test_vectors = []
    vectorizers = []
    scalers = []

    for index, train_set in enumerate(train_sets):
        logging.debug('generate_compound_model(): Transforming test set')
        vec = train_set[2]
        scale = train_set[3]
        Xte = test[columns[index]].apply(rejoin)
        Xte = vec.transform(Xte)
        Xte = scale.transform(Xte)
        test_vectors.append(Xte)

        vectorizers.append(vec)
        scalers.append(scale)
        logging.debug('generate_compound_model(): Training model ' + str(index))
        models[index].fit(train_set[0], train_set[1])

    logging.debug("generate_compound_model(): Dumping compound model to '../data/models/compound_model.pkl'")
    joblib.dump(models, '../data/models/compound_model.pkl')
    logging.debug("generate_compound_model(): Dumping vectorizers to '../data/models/compound_vectorizers.pkl'")
    joblib.dump([vectorizers, scalers], '../data/models/compound_vectorizers.pkl')

    predictions = []
    for i in range(5):
        predictions.append(models[i].predict(test_vectors[i]))

    consensus = np.array(predictions)
    consensus = stats.mode(consensus)[0]
    consensus = pd.DataFrame(consensus).T

    print(classification_report(test['class'], consensus))
    return models, test['class'], consensus


def compound_tfidf(df, column):
    logging.debug('compound_tfidf(): Transforming column: ' + column)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df[column].apply(rejoin))
    Y_train = df['class']
    s = StandardScaler(with_mean=False)
    X_train = s.fit_transform(X_train)
    return X_train, Y_train, vectorizer, s
