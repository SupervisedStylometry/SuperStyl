import jagen_will.preproc.select as sel


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', action="store", help="path to feats csv file",
                        default="feats_tests.csv")
    parser.add_argument('-s', action="store", help="optional path to already existing split file",
                        default=False)
    parser.add_argument('-m', action="store", help="path to metadata file", required=False)
    parser.add_argument('-e', action="store", help="path to excludes file", required=False)
    parser.add_argument('--lang', action="store", help="analyse only file in this language (optional, for initial split only)", required=False)
    parser.add_argument('--nosplit', action="store_true", help="no split (do not provide split file)", default=False)
    args = parser.parse_args()

    if args.nosplit:
        sel.read_clean(path=args.path,
                             metadata_path=args.m,
                             excludes_path=args.e,
                             savesplit="split_nosplit.json"
                             )
    else:

        if not args.s:
            # to create initial selection
            sel.read_clean_split(path=args.path,
                             metadata_path=args.m,
                             excludes_path=args.e,
                             savesplit="split.json",
                             lang=args.lang
                             )

        else:
            # to load and apply a selection
            sel.apply_selection(path=args.path, presplit_path=args.s)
