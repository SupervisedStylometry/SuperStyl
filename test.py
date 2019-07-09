from jagen_will.tagger import WillHelmsDeep
from jagen_will.dataset import DatasetIterator
import sys



tagger = WillHelmsDeep.load(sys.argv[1], device="cuda")

with open("test.csv", "w") as f:
    for result in tagger.test(csv_features=sys.argv[2], batch_size=4):
        f.write(result + "\n")
