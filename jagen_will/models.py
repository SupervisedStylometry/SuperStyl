import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


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


class ConvEmbedding(Encoder):
    def __init__(self,
                 input_dim, device,
                 emb_dim=128, conv_dim=256,
                 kernel_heights=None,
                 n_layers=1,  # For later
                 stride=1, padding=0,
                 dropout_ratio=0.25,
                 weights=None,
                 **kwargs):
        super().__init__(input_dim=input_dim, device=device)

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """
        print(kwargs)
        self.in_channels = 1  # This is the dimension because we have one document per one document
        self.out_channels = conv_dim

        self.kernel_heights = [5, 4, 3] if kernel_heights is None else kernel_heights

        self.stride = stride
        self.padding = padding
        self.vocab_size = input_dim
        self.embedding_length = emb_dim

        self.word_embeddings = nn.Embedding(input_dim, self.embedding_length, padding_idx=0)

        if weights:  # If we have pretrained glove or anything
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)

        self.conv = nn.ModuleList([
            nn.Conv2d(
                self.in_channels, self.out_channels,
                (kernel, self.embedding_length),
                stride, padding
            )
            for kernel in self.kernel_heights
        ])

        self.dropout = nn.Dropout(dropout_ratio)

    @property
    def params(self):
        return {
            "emb_dim": self.emb_dim,
            "conv_dim": self.emb_dim,
            "kernel_heights": self.kernel_heights,
            "dropout_ratio": self.dropout_ratio,
            "n_layers": self.n_layers,
            "stride": self.strie,
            "padding": self.padding,

        }

    @property
    def output_dimension(self):
        return len(self.kernel_heights) * self.out_channels

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input_sentences, batch_size=None):
        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out = []
        for conv in self.conv:
            max_out.append(self.conv_block(input, conv))

        all_out = torch.cat(max_out, 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)

        return fc_in


class OldConvEmbedding(Encoder):
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
        # input_dim = taille du dico
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim, padding_idx=0)

        self.linear = nn.Linear(self.emb_dim, self.emb_dim)
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

        embedded = self.linear(embedded)

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
            "second_dim": self.emb_dim,
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
