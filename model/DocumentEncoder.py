import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import init

import math

class DocumentEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, device):
        super(DocumentEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn = nn.Parameter(torch.FloatTensor(2 * hidden_dim))

        # self.pre_attn = nn.Sequential(nn.Dropout(p=dropout_rate),
        #                               nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        #                               nn.Tanh())
        # self.context_vector = nn.Parameter(torch.randn((2 * hidden_dim, 1)))
        # stdv = 1. / math.sqrt(self.context_vector.size(0))
        # self.context_vector.data.normal_(mean=0, std=stdv)
        self.device = device
        init.uniform_(self.attn.data, -0.005, 0.005)

    def forward(self, x, len_x):
        packed_seq = pack_padded_sequence(x, len_x, batch_first=True)
        out, _ = self.encoder(packed_seq)
        pad_packed_states, _ = pad_packed_sequence(out, batch_first=True)
        # pre_attn = self.pre_attn(pad_packed_states)

        # Masked attention
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=self.device)[None, :] < len_x[:, None]
        # mask = mask.unsqueeze(2)

        scores = torch.tanh(pad_packed_states.matmul(self.attn))
        # scores[~mask] = float('-inf')
        scores = torch.softmax(scores, dim=1)
        masked_scores = scores * mask.float()
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        document_embed = torch.mul(pad_packed_states, scores.unsqueeze(2).expand_as(pad_packed_states))
        document_embed = torch.sum(document_embed, dim=1)

        # dot_product = pre_attn @ self.context_vector
        # dot_product[~mask] = float('-inf')
        # attn_weights = torch.softmax(dot_product, dim=1)
        # document_embed = torch.sum(attn_weights.expand_as(pad_packed_states) * pad_packed_states, dim=1)

        return document_embed, scores

