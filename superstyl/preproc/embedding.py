import numpy as np
from scipy import spatial

def load_glove_embeddings(path):
    """
    Load GloVe embeddings from a txt file and return a dictionary mapping words to vectors.
    :param path: the path to the embeddings txt file
    :return a dictionary of words and vectors
    """
    # path = "glove.6B.100d.txt"
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def find_similar_words(embeddings_dict, word, topn=10):
    """
    Find and return the most similar words to a given word based on cosine similarity.
    :param embeddings_dict: the dictionnary of embeddings
    :param word: the word for which closest are retrieved
    :param topn: the n closest (as per cosine similarity) words on which to compute relative frequency
    :return a list of the topn closest words (including original word itself)
    """
    if word not in embeddings_dict:
        return None

    else:
        similarities = {}
        target_embedding = embeddings_dict[word]
        for other_word, other_embedding in embeddings_dict.items():
            similarity = 1 - spatial.distance.cosine(target_embedding, other_embedding)
            similarities[other_word] = similarity
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return [s[0] for s in sorted_similarities[0:topn]]

# For tests
# myTexts = [{'name': 'Letter1', 'aut': 'Smith', 'text': 'This is the text', 'lang': 'en', 'wordCounts': {'the': 1, 'this': 1}},
#  {'name': 'Letter2', 'aut': 'Smith', 'text': 'Also the text', 'lang': 'en',
#   'wordCounts': {'the': 1, 'also': 1}}]
# feat_list = ['the']
# feat = "the"
def get_embedded_counts(myTexts, feat_list, embeddings_dict, topn=10):
    """
    Replace absolute frequencies by frequencies relative to a given semantic neighbouring
    (i.e., some sort of relative frequency among 'paronyms'), using a Glove embedding (cf. Eder, 2022).
    :param myTexts: the document collection
    :param feat_list: a list of features to be selected
    :param embeddings_dict: the dictionnary of embeddings
    :param topn: the n closest (as per cosine similarity) words on which to compute relative frequency
    :return: the myTexts collection with, for each text, a 'wordCounts' dictionary with said semantic relative frequencies
    """

    for feat in feat_list:
        similars = find_similar_words(embeddings_dict, feat, topn=topn)
        if similars is None:
            # IN THAT CASE, we do not include it in the embedded freqs
            continue

        else:
            for i in enumerate(myTexts):

                if feat in myTexts[i[0]]["wordCounts"].keys():

                    if "embedded" not in myTexts[i[0]].keys():
                        # then, initialise
                        myTexts[i[0]]["embedded"] = {}

                    total = sum([myTexts[i[0]]["wordCounts"][s] for s in similars if s in myTexts[i[0]]["wordCounts"].keys()])
                    myTexts[i[0]]["embedded"][feat] = myTexts[i[0]]["wordCounts"][feat] / total


    return myTexts




