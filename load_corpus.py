from superstyl.load import load_corpus
import json

# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', nargs='+', help="paths to files", required=True)
    parser.add_argument('-o', action='store', help="optional base name of output files", type=str, default=False)
    parser.add_argument('-f', action="store", help="optional list of features in json", default=False)
    parser.add_argument('-t', action='store', help="types of features (words, chars, affixes - "
                                                   "as per Sapkota et al. 2015 - or pos). pos are currently"
                                                   "only implemented for Modern English", type=str,
                        default="words", choices=["words", "chars", "affixes", "pos"])
    parser.add_argument('-n', action='store', help="n grams lengths (default 1)", default=1, type=int)
    parser.add_argument('-k', action='store', help="How many most frequent?", default=5000, type=int)
    parser.add_argument('--freqs', action='store', help="relative, absolute or binarised freqs",
                        default="relative",
                        choices=["relative", "absolute", "binary"]
                        )
    parser.add_argument('-x', action='store', help="format (txt, xml or tei) WARNING: only txt is fully implemented",
                        default="txt",
                        choices=["txt", "xml", "tei"]
                        )
    parser.add_argument('--sampling', action='store_true', help="Sample the texts?", default=False)
    parser.add_argument('--sample_units', action='store', help="Units of length for sampling "
                                                               "(words, verses; default: words)",
                        choices=["words", "verses"],
                        default="words", type=str)
    parser.add_argument('--sample_size', action='store', help="Size for sampling (default: 3000)", default=3000, type=int)
    parser.add_argument('--sample_step', action='store', help="Step for sampling with overlap (default is no overlap)", default=None, type=int)
    parser.add_argument('--max_samples', action='store', help="Maximum number of (randomly selected) samples per class, e.g. author (default is all)",
                        default=None, type=int)
    parser.add_argument('--samples_random', action='store_true',
                        help="Should random sampling with replacement be performed instead of continuous sampling (default: false)",
                        default=False)
    parser.add_argument('--keep_punct', action='store_true', help="whether to keep punctuation and caps (default is False)",
                        default=False)
    parser.add_argument('--keep_sym', action='store_true',
                        help="if true, same as keep_punct, plus no Unidecode, and numbers are kept as well (default is False)",
                        default=False)
    parser.add_argument('--identify_lang', action='store_true',
                        help="if true, should the language of each text be guessed, using langdetect (default is False)",
                        default=False)
    parser.add_argument('--embedding', action="store", help="optional path to a word2vec embedding in txt format to compute frequencies among a set of semantic neighbourgs (i.e., pseudo-paronyms)",
                        default=False)
    parser.add_argument('--neighbouring_size', action="store", help="size of semantic neighbouring in the embedding (n closest neighbours)",
                        default=10, type=int)

    args = parser.parse_args()

    if args.f:
        print(".......loading preexisting feature list.......")
        with open(args.f, 'r') as f:
            my_feats = json.loads(f.read())

    else:
        my_feats = None

    corpus, my_feats = load_corpus(args.s, feat_list=my_feats, feats=args.t, n=args.n, k=args.k,
                                   freqsType=args.freqs, format=args.x,
                                   sampling=args.sampling, units=args.sample_units,
                                   size=args.sample_size, step=args.sample_step, max_samples=args.max_samples,
                                   samples_random=args.samples_random,
                                   keep_punct=args.keep_punct, keep_sym=args.keep_sym, identify_lang=args.identify_lang,
                                   embedding=args.embedding, neighbouring_size=args.neighbouring_size
                                   )

    print(".......saving results.......")

    if args.o:
        feat_file = args.o + "_feats.json"
        corpus_file = args.o + ".csv"

    else:
        feat_file = "feature_list_{}{}grams{}mf.json".format(args.t, args.n, args.k)
        corpus_file = "feats_tests_n{}_k_{}.csv".format(args.n, args.k)

    if not args.f:
        with open(feat_file, "w") as out:
            out.write(json.dumps(my_feats, ensure_ascii=False, indent=0))
            print("Features list saved to " + feat_file)

    corpus.to_csv(corpus_file)
    print("Corpus saved to " + corpus_file)


