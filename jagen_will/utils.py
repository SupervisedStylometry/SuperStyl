import gzip
import uuid
from contextlib import contextmanager
import os
import shutil


import torch
import torch.cuda

from typing import Dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# What follows comes from the nice https://github.com/emanjavacas/pie/blob/master/pie/utils.py
@contextmanager
def tmpfile(parent='/tmp/'):
    fid = str(uuid.uuid1())
    tmppath = os.path.join(parent, fid)
    yield tmppath
    if os.path.isdir(tmppath):
        shutil.rmtree(tmppath)
    else:
        os.remove(tmppath)


def add_gzip_to_tar(string, subpath, tar):
    with tmpfile() as tmppath:
        with gzip.GzipFile(tmppath, 'w') as f:
            f.write(string.encode())
        tar.add(tmppath, arcname=subpath)


def get_gzip_from_tar(tar, fpath):
    return gzip.open(tar.extractfile(fpath)).read().decode().strip()


def ensure_ext(path, ext, infix=None):
    """
    Compute target path with eventual infix and extension
    >>> ensure_ext("model.pt", "pt", infix="0.87")
    'model-0.87.pt'
    >>> ensure_ext("model.test", "pt", infix="0.87")
    'model-0.87.test.pt'
    >>> ensure_ext("model.test", "test", infix="pie")
    'model-pie.test'
    """
    path, oldext = os.path.splitext(path)

    # normalize extension
    if ext.startswith("."):
        ext = ext[1:]
    if oldext.startswith("."):
        oldext = oldext[1:]

    # infix
    if infix is not None:
        path = "-".join([path, infix])

    # add old extension if not the same as the new one
    if oldext and oldext != ext:
        path = '.'.join([path, oldext])

    return '.'.join([path, ext])


class Vocabulary:
    def __init__(self):
        self.unknown = "UNKNOWN"

        self.id_to_class: Dict[int, str] = {0: self.unknown}
        self.class_to_id: Dict[str, int] = {self.unknown: 0}

    def record(self, classname: str):
        if classname not in self.class_to_id:
            _id = len(self.class_to_id)
            self.class_to_id[classname] = _id
            self.id_to_class[_id] = classname

    def get_id(self, classname: str):
        return self.class_to_id.get(classname, 0)

    def get_classname(self, id_):
        return self.id_to_class[id_]

    @classmethod
    def load(cls, class_to_id: Dict[str, int]):
        self = cls()
        self.class_to_id.update(class_to_id)
        self.id_to_class.update({value: key for key, value in self.class_to_id})
        return self

    def save(self):
        return self.class_to_id

    def __len__(self):
        return len(self.class_to_id)
