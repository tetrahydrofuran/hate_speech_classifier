from . import *

def model_roulette(dataframe,
                   columns = ['tweet', 'bigram', 'trigram', 'pos', 'bigram_pos', 'trigram_pos'],
                   size=0.3, state=42):
    list_models = []
    for column in columns:
        clf, test_feat, test_result = naive_bayes.train_naive_bayes(dataframe, column, size, state)
        list_models.append(clf)
        # TODO run classification and predict etc. to do the thing

        # Trees take too long
        # list_models.append(trees.train_decision_tree(dataframe, column, size, state))