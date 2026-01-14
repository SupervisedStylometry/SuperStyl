import pandas
import csv
import random
import json
from typing import Optional, List, Tuple

def _load_metadata(path: str, metadata_path: Optional[str], 
                   excludes_path: Optional[str], lang: Optional[str]) -> Tuple[Optional[pandas.DataFrame], Optional[List[str]]]:
    """
    Load metadata and exclusion list if needed.
    """
    metadata = None
    excludes = None
    
    if metadata_path is None and (excludes_path is not None or lang is not None):
        data = pandas.read_csv(path)
        metadata = pandas.DataFrame(
            index=data.loc[:, "Unnamed: 0"],
            columns=['lang'],
            data=list(data.loc[:, "lang"])
        )
    elif metadata_path is not None:
        data = pandas.read_csv(metadata_path)
        metadata = pandas.DataFrame(
            index=data.loc[:, "id"],
            columns=['lang'],
            data=list(data.loc[:, "true"])
        )
    
    if excludes_path is not None:
        excludes_data = pandas.read_csv(excludes_path)
        excludes = list(excludes_data.iloc[:, 0])
    
    return metadata, excludes


def _should_exclude(line_id: str, metadata: Optional[pandas.DataFrame], 
                    excludes: Optional[List[str]], lang: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Determine if a line should be excluded.
    """
    if lang is not None and metadata is not None:
        try:
            if metadata.loc[line_id, "lang"] != lang:
                return True, f"not in: {lang} {line_id}"
        except KeyError:
            pass
    
    if excludes is not None and line_id in excludes:
        return True, f"Is a Wilhelmus instance! : {line_id}"
    
    return False, None


def read_clean(path: str, metadata_path: Optional[str] = None, 
               excludes_path: Optional[str] = None, 
               savesplit: Optional[str] = None, 
               lang: Optional[str] = None,
               split: bool = False,
               split_ratio: float = 0.1) -> None:
    """
    Read a CSV, clean it, and optionally split it into train and validation sets.
    
    :param path: path to CSV file
    :param metadata_path: path to metadata file (optional)
    :param excludes_path: path to file with list of excludes (optional)
    :param savesplit: path to save selection JSON (optional)
    :param lang: only include texts in this language (optional)
    :param split: if True, split into train/valid sets (default False)
    :param split_ratio: ratio for validation set when split=True (default 0.1 = 10%)
    :return: saves to disk
    """
    metadata, excludes = _load_metadata(path, metadata_path, excludes_path, lang)
    
    base_path = path.rsplit(".", 1)[0]
    
    # Initialize selection tracking
    selection = {'train': [], 'elim': []}
    if split:
        selection['valid'] = []
    
    # Open output files
    if split:
        train_path = f"{base_path}_train.csv"
        valid_path = f"{base_path}_valid.csv"
        trainf = open(train_path, 'w')
        validf = open(valid_path, 'w')
    else:
        train_path = f"{base_path}_selected.csv"
        trainf = open(train_path, 'w')
        validf = None
    
    try:
        with open(path, "r") as f:
            header = f.readline()
            trainf.write(header)
            if validf:
                validf.write(header)
            
            train_writer = csv.writer(trainf)
            valid_writer = csv.writer(validf) if validf else None
            
            print("....evaluating each text.....")
            reader = csv.reader(f, delimiter=",")
            
            for line in reader:
                line_id = line[0]
                
                # Check exclusion
                is_excluded, reason = _should_exclude(line_id, metadata, excludes, lang)
                if is_excluded:
                    selection['elim'].append(line_id)
                    if reason:
                        print(reason)
                    continue
                
                # Route to train or valid
                if split and random.random() < split_ratio:
                    selection['valid'].append(line_id)
                    valid_writer.writerow(line)
                else:
                    selection['train'].append(line_id)
                    train_writer.writerow(line)
    
    finally:
        trainf.close()
        if validf:
            validf.close()
    
    # Save selection
    if savesplit:
        with open(savesplit, "w") as out:
            out.write(json.dumps(selection))

def apply_selection(path, presplit_path):
    """
    Apply an already existing selection
    :param path: path to csv file
    :param presplit_path: path to json with selection
    :return: writes both splits on disk
    """

    with open(presplit_path, "r") as f:
        presplit = json.loads(f.read())

    trainf = open(path.split(".")[0] + "_train.csv", 'w')
    validf = open(path.split(".")[0] + "_valid.csv", 'w')


    with open(path, "r") as f:
        head = f.readline()
        trainf.write(head)
        validf.write(head)

        # and prepare to write csv lines to them
        train = csv.writer(trainf)
        valid = csv.writer(validf)

        print("....evaluating each text.....")

        reader = csv.reader(f, delimiter=",")

        for line in reader:

            if line[0] in presplit['elim']:
                print("eliminating " + line[0])
                continue

            if line[0] in presplit['valid']:
                valid.writerow(line)
                continue

            if line[0] in presplit['train']:
                train.writerow(line)

    trainf.close()
    validf.close()
