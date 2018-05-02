import nltk
import re
import logging
import string

def process_text(incoming):
    # TODO: Logging
    pass


# region Pre-tokenizing Workflow
def remove_mentions(element):
    return re.sub(r'@[a-zA-Z_0-9]{1,15}', '', element)


def remove_retweets(element):
    return re.sub(r'RT ?:?', '', element)


# region Feature extraction and removal
def emoji_extraction(text):
    return re.findall(r'&#\d+;?', text)


def emoji_removal(text):
    return re.sub(r'&#\d+;?', '', text)


def hashtag_extraction(text):
    return re.findall(r'#\w+', text)


def hashtag_removal(text):
    return re.sub(r'#\w+', '', text)
# endregion


def clean_special_characters(text):
    text = re.sub(r'&amp', 'and', text)
    text = re.sub(r'http[a-zA-Z0-9:/.-]+', '', text)  # Remove hyperlinks
    # Remove punctuation
    punct = re.compile('[{}]'.format(re.escape(string.punctuation)))
    text = re.sub(punct, '', text)
    return text
# endregion


# def tokenize(text):  # TODO: potentially unnecessary
#     return nltk.tokenize.PunktSentenceTokenizer().tokenize(text)


# region Tokenizing, n-gram creation
def tokenize(corpus):  # or just word tokenizing
    return nltk.WhitespaceTokenizer().tokenize(corpus)

def bigram_creation(corpus):
    return list(zip(corpus, corpus[1:]))


def trigram_creation(corpus):
    return list(zip(corpus, corpus[1:], corpus[2:]))
# endregion

# region Post-tokenizing Workflow
def generate_corpus(text):
    pass



def stopword_removal(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in text if word not in stopwords]


def case_correction(text):
    pass


def spelling_normalization(text):
    pass
# endregion
