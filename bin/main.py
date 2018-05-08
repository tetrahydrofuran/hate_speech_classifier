import logging
import numpy as np
import os

import pandas as pd

from processing import *
from modeling import *

logging.getLogger().setLevel(logging.DEBUG)

# region CONFIG SETTINGS
TEST_SIZE = 0.3  # Size of test set
RANDOM_STATE = 42  # np.random.randint(99999)  # random seed
REPROCESS = False
# endregion


def main():
    tweets = pd.read_csv('../data/data.csv', index_col=0)
    tweets = normalize.process_tweets(tweets, reprocess=REPROCESS)
    trees.train_decision_tree(tweets, 'tweet', TEST_SIZE, RANDOM_STATE)




if __name__ == "__main__":
    main()
