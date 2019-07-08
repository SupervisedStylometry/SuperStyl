import pandas
import tqdm
import csv

def read_clean_split(path, metadata_path=None, excludes_path=None, presplit=None, savesplit=None):
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

    train = open(path.split(".")[0] + "_train.csv", 'w')
    valid = open(path.split(".")[0] + "_valid.csv", 'w')

    selection = {'train': [], 'valid': [], 'elim': []}

    metadata = pandas.read_csv(metadata_path)

    metadata = pandas.DataFrame(index=metadata.loc[:, "id"], columns=['lang'], data=list(metadata.loc[:, "true"]))

    excludes = pandas.read_csv(excludes_path)


    with open(path, "r") as f:
        head = f.readline()
        train.write(head)
        valid.write(head)
        print("....evaluating each text.....")

        reader = csv.reader(f, delimiter=",")

        for line in tqdm.tqdm(reader):

            # First check if good language
            if not metadata.loc[line[0],"lang"] == 'nl':
                selection['elim'].append(line[0])
                # if not, eliminate it, and go to next line
                continue









    train.close()
    valid.close()