## How to use

You will need python3.6, virtualenv and pip

### Install

```bash
git clone https://github.com/Jean-Baptiste-Camps/willhelmus.git
cd willhelmus
virtualenv -p python3.6 env
source env/bin/activate
pip install -r requirements.txt
```

### Train

```bash

```

### Test

```bash

```


## If you are here, you already know.

### Get feats

```bash
python main.py [-f feature_list.json] -t chars -n 3 -s path/to/docs/*
# eg
python main.py -f feature_list.json -t chars -n 3 -s meertens-song-collection-DH2019/train/*
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

PS: if you need to download `feats.csv`, it's here: https://mab.to/GY9CfiqcD (No it sucks).

- Fasttext model that way: https://fasttext.cc/docs/en/language-identification.html

