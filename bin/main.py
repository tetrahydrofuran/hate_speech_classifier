import logging

import pandas as pd

from modeling import compound_model
from modeling import model_roulette
from processing import *

logging.getLogger().setLevel(logging.DEBUG)

# region CONFIG SETTINGS
TEST_SIZE = 0.3  # Size of test set
RANDOM_STATE = 42  # np.random.randint(99999)  # random seed
REPROCESS = False
BINARY_CLASSIFICATION = False
LABEL = 'pca-250'
GENERATE_SOLO_MODELS = True
COMPOUND_MODEL = False
DO_PCA = True
PCA_SIZE = 250
# endregion


def main():

    # Extract
    tweets = pd.read_csv('../data/data.csv', index_col=0)
    tweets = normalize.process_tweets(tweets, reprocess=REPROCESS)

    # Transform and Load
    if BINARY_CLASSIFICATION:
        tweets['class'] = tweets['class'].apply(normalize.make_binary, args=(1, 0))
    if GENERATE_SOLO_MODELS:
        model_roulette.model_roulette(tweets, label=LABEL, size=TEST_SIZE, state=RANDOM_STATE,
                                      binary=BINARY_CLASSIFICATION)
    if COMPOUND_MODEL:
        compound_model.generate_compound_model(tweets, TEST_SIZE, RANDOM_STATE)


if __name__ == "__main__":
    main()
