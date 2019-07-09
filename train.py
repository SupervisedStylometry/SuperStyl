from jagen_will.dataset import DatasetIterator
from jagen_will.tagger import WillHelmsDeep
from jagen_will import utils

if __name__ == "__main__":
    vocab = utils.Vocabulary()
    from sys import argv

    model_name = "here.model.tar"
    test = False
    if len(argv) > 1:
        model_name = argv[1]
    if len(argv) > 2:
        if argv[2] == "test":
            test = True

    if test == True:
        train = DatasetIterator(vocab, "data/train.csv")
        dev = DatasetIterator(vocab, "data/dev.csv")
    else:
        train = DatasetIterator(vocab, "data/full_feats_train.csv")
        dev = DatasetIterator(vocab, "data/full_feats_valid.csv")

    tagger = WillHelmsDeep(
        nb_features=train.nb_features,
        nb_classes=len(vocab),
        encoder_class="cnn_embedding",
        classifier_class="linear",
        classes_map=vocab,
        device="cuda",
        encoder_params=dict(emb_dim=64, hid_dim=128, n_layers=2, kernel_size=3, dropout_ratio=0.25),
        classifier_params=dict()
    )

    print(tagger.device)
    print(tagger.model)

    tagger.train(train, dev, model_name, batch_size=4, lr=1e-4, nb_epochs=50)
