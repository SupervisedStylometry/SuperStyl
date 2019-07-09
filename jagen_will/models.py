import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Dict, List, Optional


class Encoder(nn.Module):
    def __init__(self, input_dim: int, device: str = "cpu"):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

    @property
    def params(self):
        return {}

    @property
    def output_dimension(self):
        raise NotImplementedError("You are missing an implementation for the Output Dimension")


class ConvStraight(Encoder):
    def __init__(self, input_dim=15000, device: str = "cpu",
                 n_layers=3, kernel_size=3, dropout_ratio=0.1,
                 **kwargs):
        super().__init__(input_dim=input_dim, device=device)

        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.input_dim,
                                              out_channels=2 * self.input_dim,
                                              kernel_size=self.kernel_size,
                                              padding=(self.kernel_size - 1) // 2)
                                    for _ in range(self.n_layers)])

        self.dropout = nn.Dropout(self.dropout_ratio)

    def to(self, device, *args, **kwargs):
        super(ConvStraight, self).to(device, *args, **kwargs)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)

    @property
    def output_dimension(self):
        return self.input_dim

    def forward(self, src):
        """

        :param src: Tensor(batch_size, input_dim)
        :return:
        """
        # conv_input = [batch size, feature_size]
        conv_input = src

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2*hid dim]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim]
            conved = self.dropout(conved)

            # set conv_input to conved for next lo`op iteration
            conv_input = conved

        # permute and convert back to emb dim
        # conved = [batch size, emb dim, nbfeatures]
        #    --> [batch_size, nb_features * hid dim]
        return conv_input

    @property
    def params(self):
        return {
            "emb_dim": self.emb_dim,
            "hid_dim": self.hid_dim,
            "kernel_size": self.kernel_size,
            "dropout_ratio": self.dropout_ratio,
            "n_layers": self.n_layers
        }


class ConvEmbedding(Encoder):
    def __init__(self, input_dim=15000, device: str = "cpu",
                 emb_dim=128, n_layers=3, kernel_size=3, dropout_ratio=0.1,
                 **kwargs):
        super().__init__(input_dim=input_dim, device=device)

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)

        self.embedding = nn.Linear(self.input_dim, self.emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.emb_dim,
                                              out_channels=2 * self.emb_dim,
                                              kernel_size=self.kernel_size,
                                              padding=(self.kernel_size - 1) // 2)
                                    for _ in range(self.n_layers)])

        self.dropout = nn.Dropout(self.dropout_ratio)

    def to(self, device, *args, **kwargs):
        super(ConvEmbedding, self).to(device, *args, **kwargs)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)

    @property
    def output_dimension(self):
        return self.input_dim * self.emb_dim

    def forward(self, src):
        """

        :param src: Tensor(batch_size, input_dim)
        :return:
        """
        # embed features
        embedded = self.embedding(src)
        # combine embeddings by elementwise summing
        embedded = self.dropout(embedded)

        # embedded = [batch size, nb_features, emb dim]
        # conv_input = [batch size, hid dim, feature_size]
        conv_input = embedded.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2*hid dim]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim]

            # set conv_input to conved for next lo`op iteration
            conv_input = conved

        # permute and convert back to emb dim
        # conved = [batch size, emb dim, nbfeatures]
        #    --> [batch_size, nb_features * hid dim]
        return conv_input.view(conv_input.size(0), -1)

    @property
    def params(self):
        return {
            "emb_dim": self.emb_dim,
            "kernel_size": self.kernel_size,
            "dropout_ratio": self.dropout_ratio,
            "n_layers": self.n_layers
        }


class Classifier(nn.Module):
    def __init__(self,
                 encoder_output_dim, nb_classes,
                 device: str = "cpu"):
        super(Classifier, self).__init__()
        self.device = device
        self.encoder_output_dim = encoder_output_dim
        self.nb_classes = nb_classes
        
    @property
    def params(self):
        return {}


class LinearDecoder(Classifier):
    """
    Simple Linear Decoder that outputs a probability distribution
    over the classes

    """

    @property
    def params(self):
        return {
            "highway_layers": 0,
            "highway_act": 'relu'
        }

    def __init__(self,
                 encoder_output_dim, nb_classes,
                 device: str = "cpu",
                 highway_layers=0, highway_act='relu'):
        super().__init__(encoder_output_dim=encoder_output_dim, nb_classes=nb_classes, device=device)
        self.device = device

        # highway
        if highway_layers > 0:
            pass
        else:
            self.highway = None

        # decoder output
        self.decoder = nn.Linear(self.encoder_output_dim, self.nb_classes)

    def forward(self, enc_outs):
        """

        :param enc_outs: Tensor(Encoder output dimension, Classes)
        :return:
        """

        if self.highway is not None:
            enc_outs = self.highway(enc_outs)

        linear_out = self.decoder(enc_outs)

        return linear_out


class GoodWillHunting(nn.Module):
    masked_only = True

    def __init__(
        self,
        encoder: Encoder,
        classifier: Classifier,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()

        self.encoder: Encoder = encoder
        self.categorizer: Classifier = classifier
        self.device = device

        assert self.encoder.device == self.device and self.categorizer.device == self.device, \
            "All devices should be the same !"

        # nll weight
        nll_weight = torch.ones(classifier.encoder_output_dim)
        self.register_buffer('nll_weight', nll_weight)

    def forward(self, src):
        """

        :param src: Tensor(Batch_size, feature length)
        :return: Tensor(Batch_size, Classes Count)
        """
        encoder_out = self.encoder(src)

        classifier_out = self.categorizer(encoder_out)

        return classifier_out

    def predict(self, src, classnames: Optional[Dict[int, str]]) -> List[str]:
        """ Predicts value for a given tensor

        :param src: tensor(batch size x sentence_length)
        :param classnames: Name of the classes in a dict
        :return: Reversed Batch
        """
        out = self(src)
        logits = torch.argmax(out, 1)

        with torch.cuda.device_of(logits):
            out_as_list = logits.tolist()

        if classnames is None:
            return out_as_list
        else:
            return [
                classnames.get(item, "WTF ?")
                for item in out_as_list
            ]

    def train_epoch(
        self,
        src, trg, criterion=None
    ):
        """

        :param src: tensor(batch size x features)
        :param trg: tensor(batch_size x classes)
        :param scorer: Scorer
        :param criterion: Loss System
        :return: tensor(batch_size x output_length)

        """
        # (batch_size x nb_classes)
        output = self(src)

        # Target needs to be each class in a vector (batch_size) and not (batch_size * nb_classes)
        loss = criterion(output, trg.view(-1))

        return loss, torch.argmax(output, 1)
