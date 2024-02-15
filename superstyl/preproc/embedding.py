import numpy as np
from scipy import spatial
import gensim.models

def load_embeddings(path):
    """
    Load w2vec embeddings from a txt file and return a dictionary mapping words to vectors.
    :param path: the path to the embeddings txt file
    :return a dictionary of words and vectors
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    return model

def find_similar_words(model, word, topn=10):
    """
    Find and return the most similar words to a given word based on cosine similarity.
    :param model: the embedding model
    :param word: the word for which closest are retrieved
    :param topn: the n closest (as per cosine similarity) words on which to compute relative frequency
    :return a list of the topn closest words
    """
    if word not in model:
        return None

    else:
        return [s[0] for s in model.most_similar(word, topn=topn)]

# For tests
# myTexts = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en', 'wordCounts': {'the': 1, 'this': 1}},
#  {'name': 'Letter2', 'aut': 'Smith', 'text': 'Also the text', 'lang': 'en',
#   'wordCounts': {'the': 1, 'also': 1}}]
# feat_list = ['the']
# feat = "the"
def get_embedded_counts(myTexts, feat_list, model, topn=10):
    """
    Replace absolute frequencies by frequencies relative to a given semantic neighbouring
    (i.e., some sort of relative frequency among 'paronyms'), using a Glove embedding (cf. Eder, 2022).
    :param myTexts: the document collection
    :param feat_list: a list of features to be selected
    :param model: the embeddings model
    :param topn: the n closest (as per cosine similarity) words on which to compute relative frequency
    :return: the myTexts collection with, for each text, a 'wordCounts' dictionary with said semantic relative frequencies
    """

    for feat in feat_list:
        similars = find_similar_words(model, feat, topn=topn)
        if similars is None:
            # IN THAT CASE, we do not include it in the embedded freqs
            continue

        else:
            for i in enumerate(myTexts):

                if feat in myTexts[i[0]]["wordCounts"].keys():

                    if "embedded" not in myTexts[i[0]].keys():
                        # then, initialise
                        myTexts[i[0]]["embedded"] = {}

                    total = sum([myTexts[i[0]]["wordCounts"][s] for s in [feat]+similars if s in myTexts[i[0]]["wordCounts"].keys()])
                    myTexts[i[0]]["embedded"][feat] = myTexts[i[0]]["wordCounts"][feat] / total

    return myTexts




