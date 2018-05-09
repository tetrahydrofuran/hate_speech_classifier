import logging

import pandas as pd

from modeling import model_roulette
from processing import *

logging.getLogger().setLevel(logging.DEBUG)

# region CONFIG SETTINGS
TEST_SIZE = 0.3  # Size of test set
RANDOM_STATE = 42  # np.random.randint(99999)  # random seed
REPROCESS = False
# endregion


def main():
    tweets = pd.read_csv('../data/data.csv', index_col=0)
    tweets = normalize.process_tweets(tweets, reprocess=REPROCESS)
    model_roulette.model_roulette(tweets, size=TEST_SIZE, state=RANDOM_STATE)




if __name__ == "__main__":
    main()
