from jagen_will.tagger import WillHelmsDeep
from jagen_will.dataset import DatasetIterator


tagger = WillHelmsDeep.load("here.model.tar", device="cuda")

with open("test.csv", "w") as f:
    for result in tagger.test(csv_features="data/feats_tests.csv", batch_size=4):
        f.write(result + "\n")
