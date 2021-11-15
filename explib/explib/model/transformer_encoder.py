import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ..util import get_grads


class TransformerEncoderModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerEncoderModel, self).__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.ntoken = ntoken

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


######################################################################
# Load and batch data
# -------------------
#


######################################################################
# This tutorial uses ``torchtext`` to generate Wikitext-2 dataset. The
# vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Starting from sequential data, the ``batchify()``
# function arranges the dataset into columns, trimming off any tokens remaining
# after the data has been divided into batches of size ``batch_size``.
# For instance, with the alphabet as the sequence (total length of 26)
# and a batch size of 4, we would divide the alphabet into 4 sequences of
# length 6:
#
# .. math::
#   \begin{bmatrix}
#   \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#   \end{bmatrix}
#   \Rightarrow
#   \begin{bmatrix}
#   \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#   \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#   \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#   \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#   \end{bmatrix}
#
# These columns are treated as independent by the model, which means that
# the dependence of ``G`` and ``F`` can not be learned, but allows more
# efficient batch processing.
#


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` function generates the input and target sequence for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#
bptt = 35


def get_batch(source, i, device):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data.to(device), target.to(device)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# @torch.no_grad()
# def evaluate(self, data_source):
#     total_loss = 0.0
#     data_source, ntokens = data_source
#     self.model.eval()
#     self.model.to(self.device)
#     correct = 0.0
#     src_mask = self.model.generate_square_subsequent_mask(bptt).to(self.device)
#     # for i in range(0, data_source.size(0) - 1, bptt):
#     #     data, targets = get_batch(data_source, i, self.device)
#     #     if data.size(0) != bptt:
#     #         src_mask = self.model.generate_square_subsequent_mask(data.size(0)).to(
#     #             self.device
#     #         )
#     #     output = self.model(data, src_mask)
#     #     output_flat = output.view(-1, ntokens)
#     #     predicted = F.softmax(output_flat, dim=1)
#     #     _, predicted_labels = torch.max(predicted, 1)

#     #     correct += (predicted_labels == targets).sum()
#     #     # print(correct)
#     # print(correct.float() / ((len(data_source) - 1) * self.batch_size))
#     # return correct.float() / ((len(data_source) - 1) * self.batch_size)
#     for i in range(0, data_source.size(0) - 1, bptt):
#         data, targets = get_batch(data_source, i, self.device)
#         if data.size(0) != bptt:
#             src_mask = self.model.generate_square_subsequent_mask(data.size(0)).to(
#                 self.device
#             )
#         output = self.model(data, src_mask)
#         output_flat = output.view(-1, ntokens)
#         total_loss += len(data) * self.loss_func(output_flat, targets).item()
#     return total_loss / (len(data_source) - 1)


@torch.no_grad()
def evaluate(self, dataloader, no_loss=False):
    dataloader = dataloader
    self.model.eval()  # Turn on the train mode
    self.model.to(self.device)
    total_loss = 0.0
    ppl_loss = 0.0
    total_len = 0
    counter = 0
    m = 0

    for batch, (data, target, seq_len) in enumerate(dataloader):
        src_mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
        output = self.model(data, src_mask)
        output_flat = output.view(-1, self.model.ntoken)
        loss = self.loss_func(output_flat, target.view(-1)).item()
        ppl_loss += seq_len * loss
        total_len += seq_len
        if self.grad_accumulate:
            loss = loss / self.accumulate_steps
        total_loss += loss
        counter += 1
        if not self.grad_accumulate or counter % self.accumulate_steps == 0:
            m += 1
        if self.full_batch:
            break

    if no_loss:
        epoch_loss = 0
    else:
        epoch_loss = total_loss / m

    return ppl_loss / total_len, epoch_loss


def calculate_noise_norm(self, epoch=0):
    """Run through the data once without training, calculating noise norm"""
    dataloader = self.train_dataloader
    self.model.to(self.device)
    epoch_loss = 0
    n = 0
    grads = None

    self.optim.zero_grad()
    for batch, (data, target, seq_len) in enumerate(dataloader):
        if n == len(dataloader) - 1:
            break
        src_mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
        output = self.model(data, src_mask)
        output_flat = output.view(-1, self.model.ntoken)
        loss = self.loss_func(output_flat, target.view(-1))

        epoch_loss += loss.item()
        n += 1

        loss.backward()
        grad = get_grads(self.model).cpu()
        # print(grad.shape)
        if grads is None:
            grads = grad
        else:
            grads = grads + grad
        self.optim.zero_grad()

    print(n)
    epoch_loss = epoch_loss / n

    torch.save(grads, self.save_path + "/noise/grad_{}_{}".format(n, epoch))
    return calculate_norms(self, grads, n, epoch=epoch)


def calculate_norms(self, total_grads, n, epoch=0):
    print("calculate norm")
    mean_grad = total_grads / n
    noise_norms = []

    m = 0
    self.optim.zero_grad()
    epoch_loss = 0

    dataloader = self.train_dataloader
    self.optim.zero_grad()
    src_mask = self.model.generate_square_subsequent_mask(bptt).to(self.device)
    for batch, (data, target, seq_len) in enumerate(dataloader):
        src_mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
        output = self.model(data, src_mask)
        output_flat = output.view(-1, self.model.ntoken)
        loss = self.loss_func(output_flat, target.view(-1))

        epoch_loss += loss.item()
        m += 1

        loss.backward()

        grad = get_grads(self.model).cpu()
        noise_norm = (grad - mean_grad).norm().item() ** 2
        # print(noise_norm)
        noise_norms.append(noise_norm)
        self.optim.zero_grad()

    print(m)
    to_save = np.asarray(noise_norms)
    print(to_save.shape)
    np.save(
        self.save_path
        + "/noise/norm_{}_{}_{}_{}_{}_{}_{}".format(
            self.model_name,
            self.dataset_name,
            self.batch_size,
            self.seed,
            self.noise_norm_train,
            self.optim_name,
            epoch,
        ),
        to_save,
    )
