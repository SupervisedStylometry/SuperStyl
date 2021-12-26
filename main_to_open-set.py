import sys
import os
import superstyl.preproc.tuyau as tuy
import superstyl.preproc.features_extract as fex
from superstyl.preproc.text_count import count_process
import fasttext
import pandas
import json
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import tqdm
# from importlib import reload
# tuy = reload(tuy)
#import json , json.dump, file et object, json.load sur des files, dumps et loads sur des str
#import json

# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?
# Z-scores, etc. ?
# Vector-length normalisation ?

# TODO: free up memory as the script goes by deleting unnecessary objects

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", help="optional list of features in json", default=False)
    parser.add_argument('-t', action='store', help="types of features (words or chars)", type=str)
    parser.add_argument('-n', action='store', help="n grams lengths (default 1)", default=1, type=int)
    parser.add_argument('-p', action='store', help="Processes to use (default 1)", default=1, type=int)
    parser.add_argument('-c', action='store', help="Path to file with metadata corrections", default=None, type=str)
    parser.add_argument('-k', action='store', help="How many most frequent?", default=5000, type=int)
    parser.add_argument('--z_scores', action='store_true', help="Use z-scores?", default=False)
    parser.add_argument('-s', nargs='+', help="paths to files")
    args = parser.parse_args()

    model = fasttext.load_model("superstyl/preproc/models/lid.176.bin")

    print(".......loading texts.......")

    if args.c:
        # "debug_authors.csv"
        correct_aut= pandas.read_csv(args.c)
        # a bit hacky. Improve later
        correct_aut.index = list(correct_aut.loc[:, "Original"])
        myTexts = tuy.load_texts(args.s, model, correct_aut=correct_aut)

    else:
        myTexts = tuy.load_texts(args.s, model)

    print(".......Saving to csv with text.......")
    # saving
    pandas.DataFrame(columns=["authors", "texts"], index=[t["name"] for t in myTexts],
                     data= {"authors": [t["aut"] for t in myTexts],
                            "texts": [t["text"] for t in myTexts]}).to_csv("openset_feats.csv")


