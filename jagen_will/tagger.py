# Generic
import os
import tarfile
import json
import random
import uuid
import csv

# Deeeeep
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim

# Engineering beauty
from typing import Dict, Any, Optional
import tqdm

# Local imports
from jagen_will.models import GoodWillHunting, ConvEmbedding, LinearDecoder, \
    ConvStraight
from jagen_will.dataset import DatasetIterator
from jagen_will import utils

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
        self._device: str = "cpu"
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
                device=self.device,
                **encoder_params
            )
        elif encoder_class == "straight":
            self.encoder = ConvStraight(
                input_dim=nb_features,
                device=self.device,
                **encoder_params
            )
        if classifier_class == "linear":
            self.classifier = LinearDecoder(
                encoder_output_dim=self.encoder.output_dimension,
                nb_classes=self.nb_classes,
                device=self.device,
                **classifier_params or {}
            )

        self.model = GoodWillHunting(
            encoder=self.encoder,
            classifier=self.classifier,
            device=self.device
        )

        self.device = device

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

        self.classifier.device = self.encoder.device = self.model.device = self.device

        self.classifier.to(self._device)
        self.encoder.to(self._device)
        self.model.to(self._device)

    def save(self, fpath: str):
        """

        :param fpath: File path
        :return:
        """

        fpath = utils.ensure_ext(fpath, 'tar', infix=None)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(fpath, 'w') as tar:

            # serialize settings
            string, path = json.dumps(self.settings), 'settings.json.zip'
            utils.add_gzip_to_tar(string, path, tar)

            string, path = self.classes_map.dumps(), 'classes.json'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize field
            with utils.tmpfile() as tmppath:
                torch.save(self.model.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

        return fpath

    @classmethod
    def load(cls, fpath="./model.tar", device=DEVICE):
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:

            # Load the WillHelmsDeep settings
            settings = json.loads(utils.get_gzip_from_tar(tar, 'settings.json.zip'))
            print(settings)
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

    @property
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

    def write_csv(self, file, row, headers=[]):
        rows = []
        if os.path.exists(file):
            with open(file) as f:
                rows = list(csv.reader(f))
                headers, rows = rows[0], rows[1:]
        with open(file, "w") as f:
            writer: csv.writer = csv.writer(f)
            writer.writerow(headers)
            if rows:
                writer.writerows(rows)
            writer.writerow(row)

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
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1
        )

        # Generates a temp file to store the best model
        fid = '/tmp/{}'.format(str(uuid.uuid1()))
        best_valid_loss = float("inf")
        train_score = float("inf")
        dev_score = float("inf")

        for epoch in range(1, nb_epochs+1):
            try:

                train_score, train_acc = self._full_epoch(
                    iterator=train_dataset,
                    optimizer=optimizer, criterion=criterion,
                    clip=None, batch_size=batch_size,
                    desc="[Epoch Training %s/%s]" % (epoch, nb_epochs)
                )
                dev_score, dev_acc = self._full_epoch(
                    iterator=dev_dataset, criterion=criterion,
                    batch_size=batch_size,
                    desc="[Epoch Dev %s/%s]" % (epoch, nb_epochs)
                )

                # Run a check on saving the current model
                best_valid_loss = self._temp_save(fid, best_valid_loss, dev_score)

                print()
                print(f'\tTrain Loss: {train_score:.3f} | Dev Loss: {dev_score:.3f}')
                print(f'\tTrain Accu: {train_acc:.3f} | Dev Accu: {dev_acc:.3f}')
                print()
                self.write_csv(
                    model_output_path+".csv",
                    [str(epoch), f"{train_score:.3f}", f"{dev_score:.3f}", f"{train_acc:.3f}", f"{dev_acc:.3f}"],
                    ["Train loss", "Dev Loss", "Train Acc", "Dev acc"]
                )
                plateau.step(dev_score, epoch=epoch)

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

        train_mode = optimizer is not None

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

        preds = []
        trues = []

        for batch_index in tqdm.tqdm(range(0, iterator.batch_count), desc=desc):
            src, trg = next(batches)
            if train_mode:
                optimizer.zero_grad()

            loss, predictions = self.model.train_epoch(
                src, trg, criterion=criterion
            )

            with torch.cuda.device_of(predictions):
                preds.extend(predictions.tolist())
                trues.extend(trg.view(-1).tolist())

            if train_mode:
                loss.backward()

                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                optimizer.step()

            epoch_loss += loss.item()

        accuracy = sum([
            int(pred == truth)
            for pred, truth in zip(preds, trues)
        ]) / len(preds)

        return epoch_loss / iterator.batch_count, accuracy

    def predict(self, csv_features):
        dataset = DatasetIterator(
            class_encoder=self.classes_map,
            file=csv_features
        )

    def test(self, csv_features, batch_size=32):
        dataset = DatasetIterator(
            class_encoder=self.classes_map,
            file=csv_features,
            test=True
        )

        batch_generator = dataset.get_epoch(
            batch_size=batch_size,
            device=self.device,
            with_filename=True
        )
        batches = batch_generator()

        preds = []
        trues = []
        full_names = []

        for batch_index in tqdm.tqdm(range(0, dataset.batch_count), desc="Testing...."):
            src, trg, names = next(batches)

            preds.extend(self.model.predict(src, classnames=None))
            full_names.extend(names)

            with torch.cuda.device_of(trg):
                trues.extend(trg.view(-1).tolist())

        accuracy = sum([
            int(pred == truth)
            for pred, truth in zip(preds, trues)
        ]) / len(preds)

        print(accuracy)

        for pred, truth, full_name in zip(preds, trues, full_names):
            yield "{name: <10}\t{status}\t{pred: <20}\t=\t{truth: <20}".format(
                name=full_name,
                status="✓" if pred == truth else "⨯",
                pred=self.classes_map.get_classname(pred),
                truth=self.classes_map.get_classname(truth))

