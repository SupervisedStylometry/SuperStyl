from superstyl.load import load_corpus
import json

# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", help="optional list of features in json", default=False)
    parser.add_argument('-t', action='store', help="types of features (words or chars)", type=str)
    parser.add_argument('-n', action='store', help="n grams lengths (default 1)", default=1, type=int)
    parser.add_argument('-k', action='store', help="How many most frequent?", default=5000, type=int)
    parser.add_argument('--absolute_freqs', action='store_true', help="switch to get absolute instead of relative freqs", default=False)
    parser.add_argument('-s', nargs='+', help="paths to files")
    parser.add_argument('-x', action='store', help="format (txt, xml or tei)", default="txt")
    parser.add_argument('--sampling', action='store_true', help="Sample the texts?", default=False)
    parser.add_argument('--sample_units', action='store', help="Units of length for sampling (words, verses; default: words)", default="words", type=str)
    parser.add_argument('--sample_size', action='store', help="Size for sampling (default: 3000)", default=3000, type=int)
    parser.add_argument('--sample_step', action='store', help="Step for sampling with overlap (default is no overlap)", default=None, type=int)
    parser.add_argument('--max_samples', action='store', help="Maximum number of (randomly selected) samples per author/class (default is all)",
                        default=None, type=int)
    parser.add_argument('--keep_punct', action='store_true', help="whether or not to keep punctuation and caps (default is False)",
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

    corpus, my_feats = load_corpus(args.s, feat_path=args.f, feats=args.t, n=args.n, k=args.k,
                                   relFreqs=not args.absolute_freqs, format=args.x,
                                   sampling=args.sampling, units=args.sample_units,
                                   size=args.sample_size, step=args.sample_step, max_samples=args.max_samples,
                                   keep_punct=args.keep_punct, keep_sym=args.keep_sym, identify_lang=args.identify_lang,
                                   embedding=args.embedding, neighbouring_size=args.neighbouring_size
                                   )

    print(".......saving results.......")

    if not args.f:
        with open("feature_list_{}{}grams{}mf.json".format(args.t, args.n, args.k), "w") as out:
            out.write(json.dumps(my_feats, ensure_ascii=False))

    corpus.to_csv("feats_tests_n{}_k_{}.csv".format(args.n, args.k))


