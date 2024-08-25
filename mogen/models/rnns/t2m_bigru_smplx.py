import os
import numpy as np
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
from transformers import AutoTokenizer, AutoModel
from transformers import logging

from ..builder import SUBMODULES

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None):
    """
    Converts lengths to a mask tensor.

    Args:
        lengths (List[int]): List of lengths.
        device (torch.device): The device on which the tensor will be allocated.
        max_len (int, optional): The maximum length. If None, the maximum length is set to the maximum value in lengths.

    Returns:
        Tensor: A tensor mask of shape (len(lengths), max_len).
    """

    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

@SUBMODULES.register_module()
class ActorAgnosticEncoder(nn.Module):
    """
    This class is an actor-agnostic encoder for encoding input features.

    Attributes:
    - skel_embedding: a linear layer for embedding the input features.
    - mu_token, logvar_token: parameters for generating the mean and log variance of the latent distribution (only if VAE is used).
    - emb_token: parameter for generating the final output (only if VAE is not used).
    - sequence_pos_encoding: a positional encoding layer for adding positional information to the input features.
    - seqTransEncoder: a transformer encoder for encoding the input features.

    Methods:
    - __init__: initializes the ActorAgnosticEncoder object with the given parameters.
    - forward: encodes the input features and returns the encoded output.
    """

    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        """
        Initializes the ActorAgnosticEncoder object with the given parameters.

        Inputs:
        - nfeats: the number of input features.
        - vae: a flag indicating whether to use a Variational Autoencoder (VAE).
        - latent_dim: the dimension of the latent space.
        - ff_size: the size of the feedforward network in the transformer.
        - num_layers: the number of layers in the transformer.
        - num_heads: the number of attention heads in the transformer.
        - dropout: the dropout rate.
        - activation: the activation function to use in the transformer.

        Outputs: None
        """
        super().__init__()
        # self.save_hyperparameters(logger=False)
        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        # Initialize the positional encoding layer
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        # Initialize the transformer encoder layer
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        # Initialize the transformer encoder
        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )
        self.vae = vae

    def forward(self, motion, motion_length, motion_mask):
        """
        Encodes the input features and returns the encoded output.

        Inputs:
        - features: a tensor of input features.
        - lengths: a list of lengths of the input features.

        Outputs: the encoded output.
        """

        if motion_length is None:
            motion_length = [len(feature) for feature in motion]

        device = motion.device

        bs, nframes, nfeats = motion.shape
        mask = lengths_to_mask(motion_length, device)

        x = motion
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]


class DistilbertEncoderBase(nn.Module):
    """
    This class is a base encoder for DistilBERT models.

    Attributes:
    - tokenizer: the tokenizer for the pre-trained DistilBERT model.
    - text_model: the pre-trained DistilBERT model.
    - text_encoded_dim: the dimension of the hidden state in the DistilBERT model.

    Methods:
    - __init__: initializes the DistilbertEncoderBase object with the given parameters.
    - train: sets the training mode for the model.
    """

    def __init__(self, modelpath: str, finetune: bool = False):
        """
        Initializes the DistilbertEncoderBase object with the given parameters.

        Inputs:
        - modelpath: the path to the pre-trained DistilBERT model.
        - finetune: a flag indicating whether to fine-tune the DistilBERT model.

        Outputs: None
        """
        super().__init__()
        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)

        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size
        self.finetune = finetune

    def train(self, mode: bool = True):
        """
        Sets the training mode for the model.

        Inputs:
        - mode: a flag indicating whether to set the model to training mode.

        Outputs: None
        """
        self.training = mode
        for module in self.children():
            # Don't put the model in
            if module == self.text_model and not self.finetune:
                continue
            module.train(mode)
        return self

    def get_last_hidden_state(self, texts: List[str], return_mask: bool = False):
        """
        Sets the training mode for the model.

        Inputs:
        - mode: a flag indicating whether to set the model to training mode.

        Outputs: None
        """
        # Tokenize the texts and convert them to tensors
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        # Pass the encoded inputs to the DistilBERT model
        output = self.text_model(**encoded_inputs.to(self.text_model.device))

        # If not returning the attention mask, return the last hidden state
        if not return_mask:
            return output.last_hidden_state

        # If returning the attention mask, return the last hidden state and the attention mask
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)

@SUBMODULES.register_module()
class DistilbertActorAgnosticEncoder(DistilbertEncoderBase):
    def __init__(
        self,
        modelpath: str,
        finetune: bool = False,
        vae: bool = True,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        """
        Initializes the DistilbertActorAgnosticEncoder object with the given parameters.

        Inputs:
        - modelpath: the path to the pre-trained DistilBERT model.
        - finetune: a flag indicating whether to fine-tune the DistilBERT model.
        - vae: a flag indicating whether to use a VAE model.
        - latent_dim: the dimension of the latent space.
        - ff_size: the size of the feedforward network in the transformer encoder.
        - num_layers: the number of layers in the transformer encoder.
        - num_heads: the number of attention heads in the transformer encoder.
        - dropout: the dropout rate.
        - activation: the activation function to use in the transformer encoder.

        Outputs: None
        """
        super().__init__(modelpath=modelpath, finetune=finetune)
        # self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encoded_dim

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(), nn.Linear(encoded_dim, latent_dim))

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        self.vae = vae

    def forward(self, text, token, device):
        text_encoded, mask = self.get_last_hidden_state(text, return_mask=True)

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            try:
                dist = torch.distributions.Normal(mu, std)
            except ValueError:
                import ipdb

                ipdb.set_trace()  # noqa
                pass
            return dist
        else:
            return final[0]

@SUBMODULES.register_module()
class T2MContrastiveModel_SMPLX(BaseModule):

    def __init__(self, motion_encoder=None, text_encoder=None, init_cfg=None):
        super().__init__()
        self.motion_encoder = ActorAgnosticEncoder(**motion_encoder)
        self.text_encoder = DistilbertActorAgnosticEncoder(**text_encoder)
        assert init_cfg['type'] == 'Pretrained'
        self.load_pretrained(init_cfg['checkpoint'])

    def encode_motion(self,
                      motion,
                      motion_length=None,
                      motion_mask=None,
                      **kwargs):
        motion_embedding = self.motion_encoder(motion, motion_length, motion_mask).loc
        return motion_embedding

    def encode_text(self, text, token=None, device=None, **kwargs):
        text_embedding = self.text_encoder(text=text, token=token, device=device).loc
        return text_embedding

    def load_pretrained(self, ckpt_path):
        import sys
        sys.path.append("./data/evaluators/smplx322") 
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        from collections import OrderedDict
        textencoder_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split(".")[0] == "textencoder":
                name = k.replace("textencoder.", "")
                textencoder_dict[name] = v
        self.text_encoder.load_state_dict(textencoder_dict, strict=True)

        motionencoder_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split(".")[0] == "motionencoder":
                name = k.replace("motionencoder.", "")
                motionencoder_dict[name] = v
        self.motion_encoder.load_state_dict(motionencoder_dict, strict=True)

        print("T2M Evaluator Model Loaded!")
