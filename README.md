# SUPERvised STYLometry

## Installing

You will need python3.8 or later, virtualenv and pip

```bash
git clone https://github.com/SupervisedStylometry/SuperStyl.git
cd SuperStyl
virtualenv -p python3.8 env
source env/bin/activate
pip install -r requirements.txt
# And get the model for language prediction
mkdir superstyl/preproc/models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P ./superstyl/preproc/models/
```

## Workflow

FIXME: look inside the scripts, or do

```bash
python main.py --help
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
python main.py -s path/to/docs/* -t chars -n 3
# with it
python main.py -s path/to/docs/* -f feature_list.json -t chars -n 3
# There are several other available options
# See --help
```

Alternatively, you can build samples out of the data, 
for a given number of verses or words:

```bash
# words from txt
python main.py -s data/psyche/train/* -t chars -n 3 -x txt --sampling --sample_units words --sample_size 1000
# verses from TEI encoded docs
python main.py -s data/psyche/train/* -t chars -n 3 -x tei --sampling --sample_units verses --sample_size 200
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

@software{Camps_SUPERvised_STYLometry_SuperStyl_2021,author = {Camps, Jean-Baptiste},doi = {...},month = {...},title = {{SUPERvised STYLometry (SuperStyl)}},version = {...},year = {2021}}


### FastText models

## FastText

If you use these models, please cite the following papers:

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models

@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}




