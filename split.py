import jagen_will.preproc.select as sel


if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        path = argv[1]
    else:
        path = "feats_tests.csv"
    # to create initial selection
    sel.read_clean_split(path=path,
                         metadata_path="langcert_revised.csv",
                         excludes_path="wilhelmus_train.csv",
                         savesplit="split.json"
                         )

    # to load and apply a selection
    sel.apply_selection(path=path, presplit_path='split.json')
