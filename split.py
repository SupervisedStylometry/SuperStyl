import jagen_will.preproc.select as sel

if __name__ == '__main__':

    # to create initial selection
    sel.read_clean_split(path="feats_tests.csv",
                         metadata_path="langcert_revised.csv",
                         excludes_path="wilhelmus_train.csv",
                         savesplit="split.json"
                         )

    # to load and apply a selection
    sel.apply_selection(path="feats_tests.csv", presplit_path='split.json')