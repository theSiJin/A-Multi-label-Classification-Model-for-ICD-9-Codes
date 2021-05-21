from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable


# CNN Neural Network
class NNModel(nn.Module):
    def __init__(self, n_output, embed_matrix, output_conv=256, k_conv=5, dropout=0.2, lmbda=0, seed=None):
        """
        parameters:
            n_output: size of output tensors. In this research, the value should be 50
            embed_matrix: pre-trained embedding matrix
            output_conv: output size of convolutional layer
            k_conv: kernel size of convolutional layer
            dropout: dropout probability
            lmbda: value of lambda which controls regularization
            seed: random seed
        """

        super(NNModel, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.n_output = n_output
        self.lmbda = lmbda
        self.embed_dim = embed_matrix.shape[1]

        self.tanh = nn.Tanh()
        self.dp = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding.from_pretrained(embed_matrix)
        self.conv = nn.Conv1d(self.embed_dim, output_conv, k_conv, padding=int(floor(k_conv/2)))
        self.fc = nn.Linear(output_conv, n_output)

        # initialize weights
        xavier_uniform_(self.conv.weight)
        xavier_uniform_(self.fc.weight)

        if self.lmbda > 0:
            self.embedding_desc = nn.Embedding.from_pretrained(embed_matrix)
            self.conv_desc = nn.Conv1d(self.embed_dim, output_conv, k_conv, padding=int(floor(k_conv/2)))
            self.fc_desc = nn.Linear(output_conv, output_conv)

            xavier_uniform_(self.conv_desc.weight)
            xavier_uniform_(self.fc_desc.weight)

    def forward(self, x, target, desc_data=None):
        # embedding layer
        x = self.embedding(x)
        x = self.dp(x)
        x = x.transpose(1, 2)

        # convolution layer
        c = self.conv(x)
        x = F.max_pool1d(self.tanh(c), kernel_size=c.size()[2])
        x = x.squeeze(dim=2)

        # a separate module for code descriptions
        if self.lmbda > 0 and desc_data is not None:
            desc_batch = self._embed_descriptions(desc_data)
            diffs = self._penalty_from_desc(target, desc_batch)
        else:
            diffs = None

        yhat = self.fc(x)
        loss = self._get_loss(yhat, target, diffs)

        return yhat, loss

    def _embed_descriptions(self, desc_data):
        """
        compute output for layers of code description
        """
        result = []
        for desc in desc_data:
            if len(desc) > 0:
                desc_tensor = Variable(torch.LongTensor(desc))
                d = self.embedding_desc(desc_tensor)
                d = d.transpose(1,2)
                d = self.conv_desc(d)
                d = F.max_pool1d(self.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                d = self.fc_desc(d)
                result.append(d)
            else:
                result.append([])
        return result

    def _penalty_from_desc(self, target, desc_batch):
        """
        compute regularization
        """
        diffs = []
        for i, di in enumerate(desc_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().numpy()

            zi = self.fc.weight[inds, :]
            di = di[inds, :]
            diff = (zi - di).mul(zi - di).mean()

            penalty = self.lmbda * diff * di.size()[0]
            diffs.append(penalty)
        return diffs

    def _get_loss(self, yhat, target, diffs=None):
        """
        compute loss
        """
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss