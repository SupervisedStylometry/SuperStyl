import sys
import os
import jagen_will.preproc.tuyau as tuy
import jagen_will.preproc.features_extract as fex
from jagen_will.preproc.text_count import count_process
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
    parser.add_argument('-x', action='store', help="format (txt or xml)", default="txt")
    args = parser.parse_args()

    model = fasttext.load_model("jagen_will/preproc/models/lid.176.bin")

    print(".......loading texts.......")

    if args.c:
        # "debug_authors.csv"
        correct_aut = pandas.read_csv(args.c)
        # a bit hacky. Improve later
        correct_aut.index = list(correct_aut.loc[:, "Original"])
        myTexts = tuy.load_texts(args.s, model, format=args.x, correct_aut=correct_aut)

    else:
        myTexts = tuy.load_texts(args.s, model, format=args.x)

    print(".......getting features.......")

    if not args.f:
        my_feats = fex.get_feature_list(myTexts, feats=args.t, n=args.n, relFreqs=True)
        if args.k > len(my_feats):
            print("K Limit ignored because the size of the list is lower ({} < {})".format(len(my_feats), args.k))
        else:
            # and now, cut at around rank k
            val = my_feats[args.k][1]
            my_feats = [m for m in my_feats if m[1] >= val]

        with open("feature_list_{}{}grams{}mf.json".format(args.t, args.n, args.k), "w") as out:
            out.write(json.dumps(my_feats))

    else:
        print(".......loading preexisting feature list.......")
        with open(args.f, 'r') as f:
            my_feats = json.loads(f.read())

    print(".......getting counts.......")

    feat_list = [m[0] for m in my_feats]
    myTexts = fex.get_counts(myTexts, feat_list=feat_list, feats=args.t, n=args.n, relFreqs=True)

    unique_texts = [text["name"] for text in myTexts]

    print(".......feeding data frame.......")

    #feats = pandas.DataFrame(columns=list(feat_list), index=unique_texts)


    # with Pool(args.p) as pool:
    #     print(args.p)
    # target = zip(myTexts, [feat_list] * len(myTexts))
        # with tqdm.tqdm(total=len(myTexts)) as pbar:
            # for text, local_freqs in pool.map(count_process, target):

    loc = {}

    for t in tqdm.tqdm(myTexts):
        text, local_freqs = count_process((t, feat_list))
        loc[text["name"]] = local_freqs
    # Saving metadata for later
    metadata = pandas.DataFrame(columns=['author', 'lang'], index=unique_texts, data =
                                [[t["aut"], t["lang"]] for t in myTexts])
    
    # Free some space before doing this...
    del myTexts

    feats = pandas.DataFrame.from_dict(loc, columns=list(feat_list), orient="index")

    # Free some more
    del loc

    print(".......applying normalisations.......")
    # And here is the place to implement selection and normalisation
    if args.z_scores:
        feat_stats = pandas.DataFrame(columns=["mean", "std"], index=list(feat_list))
        feat_stats.loc[:,"mean"] = list(feats.mean(axis=0))
        feat_stats.loc[:, "std"] = list(feats.std(axis=0))
        feat_stats.to_csv("feat_stats.csv")

        for col in list(feats.columns):
            feats[col] = (feats[col] - feats[col].mean()) / feats[col].std()

        # TODO: vector-length normalisation?

    print(".......saving results.......")
    # frequence based selection
    # WOW, pandas is a great tool, almost as good as using R
    # But confusing as well: boolean selection works on rows by default
    # were elsewhere it works on columns
    # take only rows where the number of values above 0 is superior to two
    # (i.e. appears in at least two texts)
    #feats = feats.loc[:, feats[feats > 0].count() > 2]

    pandas.concat([metadata, feats], axis=1).to_csv("feats_tests_n{}_k_{}.csv".format(args.n, args.k))


