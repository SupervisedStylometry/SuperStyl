# Generic
import os
import tarfile
import json
import random
import uuid

# Deeeeep
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim

# Engineering beauty
from typing import Dict, Any, Optional
import tqdm

# Local imports
from .models import GoodWillHunting, ConvEmbedding, LinearDecoder
from . import utils

DEVICE = utils.DEVICE


class WillHelmsDeep:
    def __init__(
            self,
            nb_features: int,
            nb_classes: int,
            encoder_class: str,
            encoder_params: Dict[str, Any],
            classifier_class: str,
            classifier_params: Dict[str, Any],
            device: str = DEVICE,
            classes_map: utils.Vocabulary = None
    ):
        self.device: str = device
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.encoder_class = encoder_class
        self.classifier_class = classifier_class

        self.classes_map: utils.Vocabulary = classes_map

        self.encoder = None
        self.classifier = None

        # Create the classes
        if encoder_class == "cnn_embedding":
            self.encoder = ConvEmbedding(
                input_dim=nb_features,
                device=device,
                **encoder_params
            )
        if classifier_class == "linear":
            self.classifier = LinearDecoder(
                encoder_output_dim=self.encoder.output_dimension,
                nb_classes=self.nb_classes,
                device=self.device,
                **classifier_params
            )

        self.model = GoodWillHunting(
            encoder=self.encoder,
            classifier=self.classifier,
            device=self.device
        )

    def save(self, file):
        pass

    def settings(self):
        return {
            "nb_features": self.nb_features,
            "nb_classes": self.nb_classes,

            "encoder_class": self.encoder_class,
            "encoder_params": self.encoder.params,

            "classifier_class": self.classifier_class,
            "classifier_params": self.classifier.params,
            "device": self.device,
        }

    @classmethod
    def load(cls, fpath="./model.tar", device=DEVICE):
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:

            # Load the WillHelmsDeep settings
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))

            # Load the author <-> id maps
            classes = utils.Vocabulary.load(json.loads(utils.get_gzip_from_tar(tar, "classes.json")))
            assert len(classes) == settings["nb_classes"], "Vocabulary size should equal nb_classes"

            settings.update({"device": device})

            obj = cls(**settings, classes_map=classes)

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
                if device == "cpu":
                    obj.model.load_state_dict(torch.load(dictpath, map_location=device))
                else:
                    obj.model.load_state_dict(torch.load(dictpath))

        obj.model.eval()

        return obj

    def train(self,
              train_dataset, dev_dataset,
              model_output_path,
              nb_epochs: int = 10, lr: float = 1e-3,
              _seed: int = 1234,
              batch_size: int = 64):
        if _seed:
            random.seed(_seed)
            torch.manual_seed(_seed)
            torch.backends.cudnn.deterministic = True

        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Generates a temp file to store the best model
        fid = '/tmp/{}'.format(str(uuid.uuid1()))
        best_valid_loss = float("inf")
        train_score = float("inf")
        dev_score = float("inf")

        for epoch in range(1, nb_epochs+1):
            try:

                train_score = self._full_epoch(
                    iterator=train_dataset,
                    optimizer=optimizer, criterion=criterion,
                    clip=None, batch_size=batch_size,
                    desc="[Epoch Training %s/%s]" % (epoch, nb_epochs)
                )
                dev_score = self._full_epoch(
                    iterator=dev_dataset, criterion=criterion,
                    batch_size=batch_size,
                    desc="[Epoch Dev %s/%s]" % (epoch, nb_epochs)
                )

                # Run a check on saving the current model
                best_valid_loss = self._temp_save(fid, best_valid_loss, dev_score)

                print(f'\tTrain Loss: {train_score:.3f} | Dev Loss: {dev_score:.3f}')
                print()

                # Advance Learning Rate if needed
                #lr_scheduler.step(dev_score)

                #if lr_scheduler.steps >= lr_patience and lr_scheduler.lr < min_lr:
                #    raise EarlyStopException()

                #if epoch == lr_grace_periode:
                #    lr_scheduler.lr_scheduler.patience = lr_patience

                #if debug is not None:
                #    debug(self.tagger)

            except KeyboardInterrupt:
                print("Interrupting training...")
                break
            #except EarlyStopException:
            #    print("Reached plateau for too long, stopping.")
            #    break

        best_valid_loss = self._temp_save(fid, best_valid_loss, dev_score)

        try:
            self.model.load_state_dict(torch.load(fid))
            print("Saving model with loss %s " % best_valid_loss)
            os.remove(fid)
        except FileNotFoundError:
            print("No model was saved during training")

        self.save(model_output_path)

        print("Saved !")

    def _temp_save(self, file_path: str, best_score: float, current_score: float) -> float:
        if current_score < best_score:
            torch.save(self.model.state_dict(), file_path)
            best_score = current_score
        return best_score

    def _full_epoch(self, iterator, batch_size, criterion=None, optimizer=None,
                    clip: Optional[float] = None,
                    desc="Going through an epoch"):

        train_mode = optimizer is None

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0

        batch_generator = iterator.get_epoch(
            batch_size=batch_size,
            device=self.device
        )
        batches = batch_generator()

        for batch_index in tqdm.tqdm(range(0, iterator.batch_count), desc=desc):
            src, trg = next(batches)

            if train_mode:
                optimizer.zero_grad()

            loss = self.model.train_epoch(
                src, trg, criterion=criterion
            )

            if train_mode:
                loss.backward()

                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / iterator.batch_count

    def predict(self):
        pass

    def test(self):
        pass
