import superstyl.preproc.features_select as fs
import json
import regex as re

#TODO: implement more types from Sapkota et al?

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", help="list of features in json (such as produced by main.py)", required=True)
    parser.add_argument('--affixes_grams', action='store_true', help="Keep affixes (space starting or ending n-grams)", default=False)
    parser.add_argument('--punct_grams', action='store_true', help="Keep n-grams containing punctuation", default=False)
    #parser.add_argument('--word-grams', action='store_true', help="Keep n-grams with word content", default=False)
    args = parser.parse_args()

    print(".......loading preexisting feature list.......")
    with open(args.f, 'r') as f:
        my_feats = json.loads(f.read())

    print(".......Filtering feature list.......")
    my_feats = fs.filter_ngrams(my_feats, affixes=args.affixes_grams, punct=args.punct_grams)

    # name the output
    outfile = re.sub(r"\.json$", "", args.f)
    if args.affixes_grams:
        outfile = outfile+"_affixes"

    if args.punct_grams:
        outfile = outfile + "_punct"

    outfile = outfile+".json"

    print(".......Writing .......")
    with open(outfile, "w") as out:
        out.write(json.dumps(my_feats))
