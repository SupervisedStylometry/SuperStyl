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
    args = parser.parse_args()

    if not args.s:
        # to create initial selection
        sel.read_clean_split(path=args.path,
                         metadata_path=args.m,
                         excludes_path=args.e,
                         savesplit="split.json"
                         )

    else:
        # to load and apply a selection
        sel.apply_selection(path=args.path, presplit_path=args.s)