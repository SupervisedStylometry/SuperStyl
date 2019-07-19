from jagen_will.dataset import DatasetIterator
from jagen_will.tagger import WillHelmsDeep
from jagen_will import utils

from argparse import ArgumentParser
from enum import Enum


if __name__ == "__main__":
    vocab = utils.Vocabulary()

    args = ArgumentParser()
    args.add_argument("filepath", action="store", help="Output file")
    args.add_argument("-t", dest="test", action="store_true", default=False,
                      help="Activate to use the test corpus (Smaller)")
    args.add_argument("-l", dest="layers", action="store", default=3, type=int, help="Layers of the CNN")
    args.add_argument("-k", dest="kernel", action="store", default="3,4,5", type=str, help="Kernels of the CNN")
    args.add_argument("-d", dest="dropout", action="store", default=0.25, type=float, help="Dropout")
    args.add_argument("-s", dest="emb_dim", action="store", default=32, type=int,
                      help="Embedding size")
    args.add_argument("-r", dest="random", action="store_true", default=False,
                      help="Randomize the batches")
    args.add_argument("-b", dest="batch", action="store", default=4, type=int,
                      help="Batch size")
    args.add_argument("-e", dest="epochs", action="store", default=20, type=int,
                      help="Epochs")
    args.add_argument("--lr", dest="lr", action="store", default=1e-4, type=float,
                      help="Learning Rate")
    args.add_argument("-f", dest="file", action="store", default="full_feats")
    args.add_argument("-n", dest="nb_features", action="store", default=0, type=int)

    args = args.parse_args()

    model_name = args.filepath
    test = args.test
    layers = args.layers
    model = "cnn_embedding"

    # If model is embedding, it needs integers
    dataset_kwargs = dict(cast_to_int=True, randomized=args.random)
    import logging
    logging.getLogger().setLevel(logging.INFO)
    if test:
        train = DatasetIterator(vocab, "data/train.csv", **dataset_kwargs)
        dev = DatasetIterator(vocab, "data/dev.csv", **dataset_kwargs)
    else:
        train = DatasetIterator(vocab, "data/" + args.file + "_train.csv", **dataset_kwargs)
        dev = DatasetIterator(vocab, "data/" + args.file + "_valid.csv", **dataset_kwargs)

    tagger = WillHelmsDeep(
        nb_features=max(train.nb_features, dev.nb_features),
        nb_classes=len(vocab),
        encoder_class=model,
        classifier_class="linear",
        classes_map=vocab,
        max_size=train.max_size,
        device="cuda",
        encoder_params=dict(
            emb_dim=args.emb_dim,
            n_layers=layers,
            kernel_heights=list(map(int, args.kernel.split(","))),
            dropout_ratio=args.dropout
        ),
        classifier_params=dict()
    )

    print(tagger.device)
    print(tagger.model)

    tagger.train(train, dev, model_name, batch_size=args.batch, lr=args.lr, nb_epochs=args.epochs)
