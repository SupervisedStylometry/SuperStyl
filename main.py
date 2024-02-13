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
    parser.add_argument('-k', action='store', help="How many most frequent?", default=5000, type=int)
    parser.add_argument('--absolute_freqs', action='store_true', help="switch to get absolute instead of relative freqs", default=False)
    parser.add_argument('--z_scores', action='store_true', help="Use z-scores?", default=False) # TODO: remove this as already covered in model training?
    parser.add_argument('-s', nargs='+', help="paths to files")
    parser.add_argument('-x', action='store', help="format (txt, xml or tei)", default="txt")
    parser.add_argument('--sampling', action='store_true', help="Sample the texts?", default=False)
    parser.add_argument('--sample_units', action='store', help="Units of length for sampling (words, verses; default: verses)", default="verses", type=str)
    parser.add_argument('--sample_size', action='store', help="Size for sampling (default: 400)", default=400, type=int)
    parser.add_argument('--sample_step', action='store', help="Step for sampling with overlap (default is no overlap)", default=None, type=int)
    parser.add_argument('--max_samples', action='store', help="Maximum number of (randomly selected) samples per author (default is all) /!\ Only with sampling",
                        default=None, type=int)
    parser.add_argument('--keep_punct', action='store_true', help="whether or not to keep punctuation and caps (default is False)",
                        default=False)
    parser.add_argument('--keep_sym', action='store_true',
                        help="if true, same as keep_punct, plus no Unidecode, and numbers are kept as well (default is False)",
                        default=False)
    parser.add_argument('--identify_lang', action='store_true',
                        help="if true, should the language of each text be guessed, using langdetect (default is False)",
                        default=False)
    args = parser.parse_args()

    print(".......loading texts.......")

    if args.sampling:
        myTexts = tuy.docs_to_samples(args.s, identify_lang=args.identify_lang, size=args.sample_size, step=args.sample_step,
                                  units=args.sample_units, feature="tokens", format=args.x,
                                      keep_punct=args.keep_punct, keep_sym=args.keep_sym, max_samples=args.max_samples)

    else:
        myTexts = tuy.load_texts(args.s, identify_lang=args.identify_lang, format=args.x, keep_punct=args.keep_punct, keep_sym=args.keep_sym)

    print(".......getting features.......")

    if not args.f:
        my_feats = fex.get_feature_list(myTexts, feats=args.t, n=args.n, relFreqs=not args.absolute_freqs)
        if args.k > len(my_feats):
            print("K Limit ignored because the size of the list is lower ({} < {})".format(len(my_feats), args.k))
        else:
            # and now, cut at around rank k
            val = my_feats[args.k][1]
            my_feats = [m for m in my_feats if m[1] >= val]

        with open("feature_list_{}{}grams{}mf.json".format(args.t, args.n, args.k), "w") as out:
            out.write(json.dumps(my_feats, ensure_ascii=False))

    else:
        print(".......loading preexisting feature list.......")
        with open(args.f, 'r') as f:
            my_feats = json.loads(f.read())

    print(".......getting counts.......")

    feat_list = [m[0] for m in my_feats]
    myTexts = fex.get_counts(myTexts, feat_list=feat_list, feats=args.t, n=args.n, relFreqs=not args.absolute_freqs)

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

        # TODO: vector-length normalisation? -> No, in pipeline

    print(".......saving results.......")
    # frequence based selection
    # WOW, pandas is a great tool, almost as good as using R
    # But confusing as well: boolean selection works on rows by default
    # were elsewhere it works on columns
    # take only rows where the number of values above 0 is superior to two
    # (i.e. appears in at least two texts)
    #feats = feats.loc[:, feats[feats > 0].count() > 2]

    pandas.concat([metadata, feats], axis=1).to_csv("feats_tests_n{}_k_{}.csv".format(args.n, args.k))


