## How to use

You will need python3.6, virtualenv and pip

### Install

```bash
git clone https://github.com/Jean-Baptiste-Camps/willhelmus.git
cd willhelmus
virtualenv -p python3.6 env
source env/bin/activate
pip install -r requirements.txt
# And get the model for language prediction
mkdir jagen_will/preproc/models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P ./jagen_will/preproc/models/
```

### Train

```bash

```

### Test

```bash

```


## If you are here, you already know.

### Get feats

With or without preexisting feature list:

```bash
python main.py -t chars -n 3 -c debug_authors.csv [-p 1] -k 5000 -s path/to/docs/*
# with it
python main.py -f feature_list.json -t chars -n 3 -c debug_authors.csv -k 5000 -s meertens-song-collection-DH2019/train/*
```


### Do the split

If you want to do initial random split,
```bash
python split.py feats_tests.csv -m langcert_revised.csv -e wilhelmus_train.csv
```

If you want to split according to existing json file,
```bash
python split.py feats_tests.csv -s split.json
```

### Train svm

It's quite simple riilly,
```bash
python train_svm.py path-to-train-data.csv path-to-test-data.csv [--norms] [--dim_reduc None, 'pca', 'som'] [--kernel, 'LinearSVC', 'linear', 'polynomial', 'rbf', 'sigmoid'] [--final]
# e.g.
python train_svm.py data/feats_tests_train.csv data/feats_tests_valid.csv --norms --dim_reduc som
```

PS: if you need to download `feats.csv`, it's here: https://mab.to/GY9CfiqcD (No it sucks).

- Fasttext model that way: https://fasttext.cc/docs/en/language-identification.html

