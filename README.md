# SUPERvised STYLometry

[![codecov](https://codecov.io/github/SupervisedStylometry/SuperStyl/graph/badge.svg?token=TY5HCBOOKL)](https://codecov.io/github/SupervisedStylometry/SuperStyl)
[![DOI](https://zenodo.org/badge/342586864.svg)](https://doi.org/10.5281/zenodo.14069799)

## Installing

You will need python3.9 or later, `pip` and optionnaly `virtualenv`

```bash
git clone https://github.com/SupervisedStylometry/SuperStyl.git
cd SuperStyl
virtualenv -p python3.9 env #or later
source env/bin/activate
pip install -r requirements.txt
```

## Basic usage

To use Superstyl, you have two options:

1. Use the provided command-line interface from your OS terminal (tested on Linux)
2. Import Superstyl in a Python script or notebook, and use the API commands

You also need a collection of files containing the text that you wish
to analyse. The naming conventions of source files in Superstyl are as such:

```
Class_anythingthatyouwant
```

For instance:
```
Moliere_Amphitryon.txt
```

The text before the first underscore will be used as the class for training models.

### Command-Line Interface

A very simple usage, for building a corpus of text character 3-grams frequencies, 
training a SVM model with leave-one-out cross-validation, 
and predicting the class of unknown texts, would be:

```bash
# Creating the corpus and extracting characters 3-grams from text files
python load_corpus.py -s data/train/*.txt -t chars -n 3 -o train
python load_corpus.py -s data/test/*.txt -t chars -n 3 -o unknown -f train_feats.json
# Training a SVM, with cross-validation, and using it to predict the class of unknown sample
python train_svm.py train.csv --test_path unknown.csv --cross_validate leave-one-out --final
```

The two first commands will write to the disk the files `train.csv` and `unknown.csv` 
containing the metadata and features frequencies for both sets of files, 
and a file `train_feats.json` containing a list of used features.

The last one will print the scores of the cross-validation, and then write 
to disk a file `FINAL_PREDICTIONS.csv`, containing the class predictions 
for the unknown texts.

This is just a small sample of all available corpus and training options.

To know more, do:
```commandline
python load_corpus.py --help
python train_svm.py --help
```

### Python API

A very simple usage, for building a corpus, training a SVM model with cross-validation, 
and predicting the class of an unknown text, would be:

```python
import superstyl as sty
import glob
# Creating the corpus and extracting characters 3-grams from text files
train, train_feats = sty.load_corpus(glob.glob("data/train/*.txt"), 
                                           feats="chars", n=3)
unknown, unknown_feats = sty.load_corpus(glob.glob("data/test/*.txt"), 
                                         feat_list=train_feats, 
                                         feats="chars", n=3)
# Training a SVM, with cross-validation, and using it 
# to predict the class of unknown sample
sty.train_svm(train, unknown, cross_validate="leave-one-out", 
              final_pred=True)
```

<!-- TODO: update when train_svm api will be modified -->


This is just a small sample of all available corpus and training options.

To know more, do:
```python
help(sty.load_corpus)
help(sty.train_svm)
```


## Advanced usage

FIXME: look inside the scripts, or do

```bash
python load_corpus.py --help
python train_svm.py --help
```

for full documentation on the main functionnalities of the CLI, regarding data generation (`main.py`) and SVM training (`train_svm.py`).

For more particular data processing usages (splitting and merging datasets), see also:

```bash
python split.py --help
python merge_datasets.csv.py --help
```


### Get feats

With or without preexisting feature list:

```bash
python load_corpus.py -s path/to/docs/* -t chars -n 3
# with it
python load_corpus.py -s path/to/docs/* -f feature_list.json -t chars -n 3
# There are several other available options
# See --help
```

Alternatively, you can build samples out of the data, 
for a given number of verses or words:

```bash
# words from txt
python load_corpus.py -s data/psyche/train/* -t chars -n 3 -x txt --sampling --sample_units words --sample_size 1000
# verses from TEI encoded docs
python load_corpus.py -s data/psyche/train/* -t chars -n 3 -x tei --sampling --sample_units verses --sample_size 200
```

You have a lot of options for feats extraction, inclusion or not of punctuation and symbols, sampling, source file formats, …, that can be accessed through the help.

### Optional: Merge different features

You can merge several sets of features, extracted in csv with the previous commands, by doing:

```bash
python merge_datasets.csv.py -o merged.csv char3grams.csv words.csv affixes.csv
```

### Optional: Do a fixed split

You can choose either choose to perform k-fold cross-validation (including leave-one-out), in which case
this step is unnecessary. Or you can do a classical train/test random split.

If you want to do initial random split,
```bash
python split.py feats_tests.csv
```

If you want to split according to existing json file,
```bash
python split.py feats_tests.csv -s split.json
```

There are other available options, see `--help`, e.g.

```bash
python split.py feats_tests.csv -m langcert_revised.csv -e wilhelmus_train.csv
```


### Train svm

It's quite simple really,

```bash
python train_svm.py path-to-train-data.csv [--test_path TEST_PATH] [--cross_validate {leave-one-out,k-fold}] [--k K] [--dim_reduc {pca}] [--norms] [--balance {class_weight,downsampling,Tomek,upsampling,SMOTE,SMOTETomek}] [--class_weights] [--kernel {LinearSVC,linear,polynomial,rbf,sigmoid}] [--final] [--get_coefs]
```

For instance, using leave-one-out or 10-fold cross-validation

```bash
# e.g.
python train_svm.py data/feats_tests_train.csv --norms --cross_validate leave-one-out
python train_svm.py data/feats_tests_train.csv --norms --cross_validate k-fold --k 10
```

Or a train/test split

```bash
# e.g.
python train_svm.py data/feats_tests_train.csv --test_path test_feats.csv --norms
```

And for a final analysis, applied on unseen data:

```bash
# e.g.
python train_svm.py data/feats_tests_train.csv --test_path unseen.csv --norms --final
```

With a little more options,

```bash
# e.g.
python train_svm.py data/feats_tests_train.csv --test_path unseen.csv --norms --class_weights --final --get_coefs
```



## Sources

## Cite this repository

You can cite it using the CITATION.cff file (and Github cite functionnalities), following:

BIBTEX:

```bibtex
@software{camps_cafiero_2024,
  author       = {Jean-Baptiste Camps and Florian Cafiero},
  title        = {{SUPERvised STYLometry (SuperStyl)}},
  month        = {11},
  year         = {2024},
  version      = {v1.0},
  doi          = {10.5281/zenodo.14069799},
  url          = {https://doi.org/10.5281/zenodo.14069799}
}
```


MLA:

```plaintext
Camps, Jean-Baptiste, and Florian Cafiero. *SUPERvised STYLometry (SuperStyl)*. Version 1.0, 11 Nov. 2024, doi:10.5281/zenodo.14069799.
```









