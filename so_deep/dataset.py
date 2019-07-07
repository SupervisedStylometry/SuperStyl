import logging
import collections
import random

# DEEEP
import torch.nn

# Engineeringgggg BEAUTY
from typing import Tuple, Dict, List, Optional, Iterator, Sequence, Callable, Union

# From here
from . import utils

GT_PAIR = collections.namedtuple("GT", ("x", "y", "line_index"))
DEVICE = utils.DEVICE


class DatasetIterator:
    def __init__(self,
                 class_encoder: utils.Vocabulary,
                 file,
                 batch_size: int = 32,
                 randomized: bool = False):

        self._class_encoder = class_encoder

        self.encoded: List[GT_PAIR] = []
        self.current_epoch: List[tuple, int] = []
        self.random = randomized
        self.file = file
        self.batch_count = 0
        self.batch_size = batch_size

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
            for line_index, line in enumerate(fio.readlines()):
                if not line.strip() or line_index == 0:
                    continue

                x, y = self.read_unit(*line.strip().split("\t"))

                self.encoded.append(
                    GT_PAIR(x, y, line_index)
                )

        logging.info("DatasetIterator found {} lines in {}".format(len(self), self.file))

        # Get the number of batch for TQDM
        self.batch_count = len(self) // self.batch_size + bool(len(self) % self.batch_size)

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_count = len(self) // self.batch_size + bool(len(self) % self.batch_size)

    def read_unit(self, name, aut, lang, *features) -> Tuple[List[float], List[int]]:
        """ Returns x then y

        :param name:
        :param aut:
        :param lang:
        :param features:
        :return:
        """
        y = self._class_encoder.get_classname(aut)
        x = features
        return list(x), [y]

    def get_epoch(self, device: str = DEVICE, batch_size: int = 32) -> Callable[[], Iterator[Tuple[torch.Tensor, ...]]]:
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

                for gt_pair in lines[n:n+self.batch_size]:
                    xs.append(gt_pair.x)
                    y_trues.append(gt_pair.y)

                try:
                    x_tensor = torch.tensor(xs, device=device, dtype=torch.int)
                    y_tensor = torch.tensor(y_trues, device=device, dtype=torch.float)
                except ValueError:
                    print([b.line_index for b in lines[n:n+self.batch_size]])
                    raise
                yield (
                    x_tensor,
                    y_tensor
                )

        return iterable
