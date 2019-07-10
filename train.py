from jagen_will.dataset import DatasetIterator
from jagen_will.tagger import WillHelmsDeep
from jagen_will import utils

from argparse import ArgumentParser
from enum import Enum


class Models(Enum):
    cnn_embedding = "cnn_embedding"
    cnn_linear = "cnn_linear"
    straight = "straight"
    cnn_no_cast = "cnn_no_cast"

if __name__ == "__main__":
    vocab = utils.Vocabulary()

    args = ArgumentParser()
    args.add_argument("filepath", action="store", help="Output file")
    args.add_argument("-t", dest="test", action="store_true", default=False,
                      help="Activate to use the test corpus (Smaller)")
    args.add_argument("-m", dest="model",
                      action="store", type=Models, choices=list(Models), default=Models.cnn_embedding,
                      help="Structure of the CNN to use")
    args.add_argument("-l", dest="layers", action="store", default=3, type=int, help="Layers of the CNN")
    args.add_argument("-k", dest="kernel", action="store", default=5, type=int, help="Kernels of the CNN")
    args.add_argument("-d", dest="dropout", action="store", default=0.25, type=float, help="Dropout")
    args.add_argument("-s", dest="size", action="store", default=32, type=int,
                      help="Size of the first layer (Embedding or Linear)")
    args.add_argument("-r", dest="random", action="store_true", default=False,
                      help="Randomize the batches")
    args.add_argument("-b", dest="batch", action="store", default=4, type=int,
                      help="Batch size")
    args.add_argument("-e", dest="epochs", action="store", default=20, type=int,
                      help="Epochs")
    args.add_argument("--lr", dest="lr", action="store", default=1e-4, type=float,
                      help="Learning Rate")
    args.add_argument("-f", dest="file", action="store", default="full_feats")

    args = args.parse_args()

    model_name = args.filepath
    test = args.test
    layers = args.layers
    model = args.model.value

    # If model is embedding, it needs integers
    cast_to_int = model == "cnn_embedding"
    dataset_kwargs = dict(cast_to_int=cast_to_int, randomized=args.random)
    if model == "cnn_no_cast":
        model = "cnn_embedding"

    if test:
        train = DatasetIterator(vocab, "data/train.csv", **dataset_kwargs)
        dev = DatasetIterator(vocab, "data/dev.csv", **dataset_kwargs)
    else:
        train = DatasetIterator(vocab, "data/" + args.file + "_train.csv", **dataset_kwargs)
        dev = DatasetIterator(vocab, "data/" + args.file + "_valid.csv", **dataset_kwargs)

    tagger = WillHelmsDeep(
        nb_features=train.nb_features,
        nb_classes=len(vocab),
        encoder_class=model,
        classifier_class="linear",
        classes_map=vocab,
        device="cpu",
        encoder_params=dict(
            second_dim=args.size,
            n_layers=layers,
            kernel_size=args.kernel,
            dropout_ratio=args.dropout
        ),
        classifier_params=dict()
    )

    print(tagger.device)
    print(tagger.model)

    tagger.train(train, dev, model_name, batch_size=args.batch, lr=args.lr, nb_epochs=args.epochs)
