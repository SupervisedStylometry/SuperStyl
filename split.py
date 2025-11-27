"""
Command-line tool for splitting datasets.
"""

import superstyl.preproc.select as sel


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', action="store", help="path to feats csv file",
                        default="feats_tests.csv")
    parser.add_argument('-s', action="store", help="optional path to already existing split file",
                        default=False)
    parser.add_argument('-m', action="store", help="path to metadata file", required=False)
    parser.add_argument('-e', action="store", help="path to excludes file", required=False)
    parser.add_argument('--lang', action="store", 
                        help="analyse only file in this language (optional, for initial split only)", 
                        required=False)
    parser.add_argument('--nosplit', action="store_true", 
                        help="no split (do not provide split file)", 
                        default=False)
    parser.add_argument('--split_ratio', action="store", type=float,
                        help="validation split ratio (default: 0.1 = 10%%)",
                        default=0.1)
    args = parser.parse_args()

    if args.s:
        # Apply existing selection
        sel.apply_selection(path=args.path, presplit_path=args.s)
    else:
        # Create new selection (with or without split)
        sel.read_clean(
            path=args.path,
            metadata_path=args.m,
            excludes_path=args.e,
            savesplit="split_nosplit.json" if args.nosplit else "split.json",
            lang=args.lang,
            split=not args.nosplit,
            split_ratio=args.split_ratio
        )
