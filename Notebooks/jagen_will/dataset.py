import logging
import collections
import random
import csv

# DEEEP
import torch.nn

# Engineeringgggg BEAUTY
from typing import Tuple, Dict, List, Optional, Iterator, Sequence, Callable, Union

# From here
from jagen_will import utils

GT_PAIR = collections.namedtuple("GT", ("x", "y", "line_index", "file_name"))
DEVICE = utils.DEVICE


def cast_to_int_fn(float_number: str) -> int:
    return int(float(float_number) * 10000)


class DatasetIterator:
    def __init__(self,
                 class_encoder: utils.Vocabulary,
                 file,
                 batch_size: int = 32,
                 randomized: bool = False,
                 test=False,
                 cast_to_int=True):

        self._class_encoder = class_encoder

        self.encoded: List[GT_PAIR] = []
        self.current_epoch: List[tuple, int] = []
        self.random = randomized
        self.file = file
        self.batch_count = 0
        self.batch_size = batch_size
        self.type = "test" if test else "train"
        self.nb_features: int = 0

        self.cast_to_int = cast_to_int

        self._setup()

    def __repr__(self):
        return "<DatasetIterator examples='{}' random='{}' \n" \
               "\t batches='{}' batch_size='{}'/>".format(
                    len(self),
                    self.random,
                    self.batch_count,
                    self.batch_size
                )

    def __len__(self):
        """ Number of examples
        """
        return len(self.encoded)

    def _setup(self):
        """ The way this whole iterator works is pretty simple :
        we look at each line of the document, and store its index. This will allow to go directly to this line
        for each batch, without reading the entire file. To do that, we need to read in bytes, otherwise file.seek()
        is gonna cut utf-8 chars in the middle
        """
        logging.info("DatasetIterator reading indexes of lines")
        with open(self.file, "r") as fio:
            reader = csv.reader(fio)
            for line_index, line in enumerate(reader):
                if not line or line_index == 0:
                    continue
                try:
                    x, y, fname = self.read_unit(*line)
                except Exception:
                    print(line)
                    raise
                self.encoded.append(
                    GT_PAIR(x, y, line_index, fname)
                )

        self.nb_features = len(self.encoded[-1].x)

        logging.info("DatasetIterator found {} lines in {}".format(len(self), self.file))

        # Get the number of batch for TQDM
        self.batch_count = len(self) // self.batch_size + bool(len(self) % self.batch_size)

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_count = len(self) // self.batch_size + bool(len(self) % self.batch_size)

    def read_unit(self, name, aut, lang, *features) -> Tuple[List[float], List[Union[int]], str]:
        """ Returns x then y

        :param name: Name of the file
        :param aut: Author class
        :param lang: Lang class
        :param features: Features
        :return: xs, [y], filename
        """
        if self.type != "test":
            self._class_encoder.record(aut)
        #print(name, aut, lang, features)
        y = self._class_encoder.get_id(aut)
        xs = features
        if self.cast_to_int:
            return list(map(cast_to_int_fn, xs)), [y], name
        return list(map(float, xs)), [y], name

    def get_epoch(self, device: str = DEVICE, batch_size: int = 32, with_filename=False) -> Callable[[], Iterator[Tuple[torch.Tensor, ...]]]:
        # If the batch size is not the original one (most probably is !)
        if batch_size != self.batch_size:
            self.reset_batch_size(batch_size)

        # Create a list of lines
        lines = [] + self.encoded

        # If we need randomization, then DO randomization shuffle of lines
        if self.random is True:
            random.shuffle(lines)

        def iterable():
            for n in range(0, len(lines), self.batch_size):
                xs, y_trues = [], []

                filenames = []

                for gt_pair in lines[n:n+self.batch_size]:
                    xs.append(gt_pair.x)
                    y_trues.append(gt_pair.y)
                    filenames.append(gt_pair.file_name)

                try:
                    x_tensor = torch.tensor(xs, device=device)
                    y_tensor = torch.tensor(y_trues, device=device)
                except ValueError:
                    print([b.line_index for b in lines[n:n+self.batch_size]])
                    raise

                if with_filename:
                    yield (x_tensor, y_tensor, filenames)
                else:
                    yield (x_tensor, y_tensor)

        return iterable


if __name__ == "__main__":
    vocab = utils.Vocabulary()
    train = DatasetIterator(vocab, "data/train.csv")
    epoch = train.get_epoch()

    batches = epoch()
    for x, y in batches:
        print(x.shape, y.shape)
