import nltk
import pandas as pd
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import logging
from gensim.models import word2vec

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3)


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence,
                                                remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

if __name__ == 'main':
    sentences = []  # Initialize an empty list of sentences

    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)



    # Initialize and train the model (this will take some time)

    model = word2vec.Word2Vec(sentences, workers=num_workers,
                size=num_features, min_count = min_word_count,
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)