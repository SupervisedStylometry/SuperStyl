import jagen_will.preproc.tuyau as tuy
import jagen_will.preproc.features_extract as fex
from jagen_will.preproc.text_count import count_process, encode_texts

import fasttext
import json
# from multiprocessing import Pool
import tqdm
import csv

# Pandas
import pandas

# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?
# Z-scores, etc. ?
# Vector-length normalisation ?

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", help="optional list of features in json", default=False)
    parser.add_argument('-t', action='store', help="types of features (words or chars)", type=str)
    parser.add_argument('-n', action='store', help="n grams lengths (default 1)", default=1, type=int)
    parser.add_argument('-p', action='store', help="Processes to use (default 1)", default=1, type=int)
    parser.add_argument('-c', action='store', help="Path to file with metadata corrections", default=None, type=str)
    parser.add_argument('--z_scores', action='store_true', help="Use z-scores?", default=False)
    parser.add_argument('-m', action='store', dest="max_size", help="Use z-scores?", default=5000, type=int)
    parser.add_argument('-s', nargs='+', help="paths to files")
    args = parser.parse_args()

    model = fasttext.load_model("jagen_will/preproc/models/lid.176.bin")

    print(".......loading texts.......")

    if args.c:
        # "debug_authors.csv"
        correct_aut = pandas.read_csv(args.c)
        # a bit hacky. Improve later
        correct_aut.index = list(correct_aut.loc[:, "Original"])
        myTexts = tuy.load_texts(args.s, model, correct_aut=correct_aut)

    else:
        myTexts = tuy.load_texts(args.s, model)

    print(".......getting features.......")

    if not args.f:
        my_feats = fex.get_feature_list(myTexts, feats=args.t, n=args.n, relFreqs=True,
                                        keys_only=True)

        # + 2 where 2 = pad + unk
        feat_map = dict(PAD=0, UNK=1, **{key: index+2 for index, key in enumerate(my_feats)})
        with open("twitter_feats.json", "w") as f:
            json.dump(feat_map, f)
    else:
        with open(args.f) as f:
            feat_map = json.load(f)

    print(".......Transcoding map..........")
    print("   => Lengths: {} ".format(len(feat_map)))
    # print(feat_map)
    print(".......feeding data frame.......")
    loc = {}
    max_size = 0
    ignored = 0
    for t in tqdm.tqdm(myTexts):
        text, transcode = encode_texts(t, feat_map, n=args.n, feats=args.t)
        size = len(transcode)
        if size < args.max_size:
            max_size = max(max_size, size)
            loc[text["name"]] = [text["aut"], text["lang"]] + list(map(str, transcode))
        else:
            ignored += 1

    print("Ignore for size = {}".format(ignored))
    print("Max size = {} ".format(max_size))
    max_size = max(map(len, loc.values()))
    print("Max size = {} ".format(max_size))

    print(".......saving results.......")
    # frequence based selection
    # WOW, pandas is a great tool, almost as good as using R
    # But confusing as well: boolean selection works on rows by default
    # were elsewhere it works on columns
    # take only rows where the number of values above 0 is superior to two
    # (i.e. appears in at least two texts)
    # feats = feats.loc[:, feats[feats > 0].count() > 2]

    with open("data/twitter_feats.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "aut", "lang"])
        for text_name, transcoded in loc.items():
            writer.writerow([text_name] + transcoded + (["0"] * (max_size - len(transcoded))))



