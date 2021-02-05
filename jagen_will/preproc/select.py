import pandas
import csv
import random
import json

# TODO: make same modifs for the no split
def read_clean_split(path, metadata_path=None, excludes_path=None, savesplit=None, lang=None):
    """
    Function to read a csv, clean it, and then split it in train and dev,
    either randomly or according to a preexisting selection
    :param path: path to csv file
    :param metadata_path: path to metadata file
    :param excludes_path: path to file with list of excludes
    :param presplit: path to file with preexisting split (optional)
    :param savesplit: path to save split (optional)
    :return: saves to disk
    """

    trainf = open(path.split(".")[0] + "_train.csv", 'w')
    validf = open(path.split(".")[0] + "_valid.csv", 'w')

    selection = {'train': [], 'valid': [], 'elim': []}

    # Do we need to create metadata ?
    if metadata_path is None and (excludes_path is not None or lang is not None):
        metadata = pandas.read_csv(path)
        metadata = pandas.DataFrame(index=metadata.loc[:, "Unnamed: 0"], columns=['lang'], data=list(metadata.loc[:, "lang"]))

    if metadata_path is not None:
        metadata = pandas.read_csv(metadata_path)
        metadata = pandas.DataFrame(index=metadata.loc[:, "id"], columns=['lang'], data=list(metadata.loc[:, "true"]))

    if excludes_path is not None:
        excludes = pandas.read_csv(excludes_path)
        excludes = list(excludes.iloc[:, 0])

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

            # checks
            if lang is not None:
                # First check if good language
                if not metadata.loc[line[0], "lang"] == lang:
                    selection['elim'].append(line[0])
                    print("not in: " + lang + " " + line[0])
                    # if not, eliminate it, and go to next line
                    continue

            if excludes_path is not None:
                # then check if to exclude
                if line[0] in excludes:
                    selection['elim'].append(line[0])
                    print("Is a Wilhelmus instance! : " + line[0])
                    # then eliminate it, and go to next line
                    continue

            # Now that we have only the good lines, proceed to split

            # 10% for dev
            if random.randint(1, 10) == 1:
                selection['valid'].append(line[0])
                valid.writerow(line)

            # 90% for train
            else:
                selection['train'].append(line[0])
                train.writerow(line)

    trainf.close()
    validf.close()
    with open(savesplit, "w") as out:
        out.write(json.dumps(selection))

def read_clean(path, metadata_path=None, excludes_path=None, savesplit=None):
    """
    Function to read a csv, clean it.
    :param path: path to csv file
    :param metadata_path: path to metadata file
    :param excludes_path: path to file with list of excludes
    :return: saves to disk
    """
    # TODO: FIX by implementing all necessary options

    trainf = open(path.split(".")[0] + "_selected.csv", 'w')

    selection = {'train': [], 'elim': []}

    metadata = pandas.read_csv(metadata_path)

    metadata = pandas.DataFrame(index=metadata.loc[:, "id"], columns=['lang'], data=list(metadata.loc[:, "true"]))

    excludes = pandas.read_csv(excludes_path)
    excludes = list(excludes.iloc[:, 0])

    with open(path, "r") as f:
        head = f.readline()
        trainf.write(head)

        # and prepare to write csv lines to them
        train = csv.writer(trainf)

        print("....evaluating each text.....")

        reader = csv.reader(f, delimiter=",")

        for line in reader:

            # First check if good language
            if not metadata.loc[line[0], "lang"] == 'nl':
                selection['elim'].append(line[0])
                print("not in dutch: " + line[0])
                # if not, eliminate it, and go to next line
                continue

            # then check if to exclude
            if line[0] in excludes:
                selection['elim'].append(line[0])
                print("Is a Wilhelmus instance! : " + line[0])
                # then eliminate it, and go to next line
                continue

            # Now that we have only the good lines, proceed to write
            selection['train'].append(line[0])
            train.writerow(line)

    trainf.close()
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
