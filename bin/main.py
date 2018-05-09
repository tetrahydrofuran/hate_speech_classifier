import logging

import pandas as pd

from modeling import model_roulette
from processing import *

logging.getLogger().setLevel(logging.DEBUG)

# region CONFIG SETTINGS
TEST_SIZE = 0.3  # Size of test set
RANDOM_STATE = 42  # np.random.randint(99999)  # random seed
REPROCESS = False
BINARY_CLASSIFICATION = True
LABEL = 'what'
# endregion


def main():
    tweets = pd.read_csv('../data/data.csv', index_col=0)
    tweets = normalize.process_tweets(tweets, reprocess=REPROCESS)
    if BINARY_CLASSIFICATION:
        tweets['class'] = tweets['class'].apply(normalize.make_binary, args=(1, 2))

    model_roulette.model_roulette(tweets, label=LABEL, size=TEST_SIZE, state=RANDOM_STATE, binary=BINARY_CLASSIFICATION)


if __name__ == "__main__":
    main()
