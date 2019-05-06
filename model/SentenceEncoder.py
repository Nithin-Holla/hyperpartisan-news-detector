import torch
from torch import nn


class SentenceEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(SentenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pre_attn = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                                      nn.Tanh())
        self.context_vector = nn.Parameter(torch.zeros((2 * hidden_dim, 1)))

    def forward(self, x):
        out, hidden = self.encoder(x)   # slice and reorder out later
        pre_attn = self.pre_attn(out)
        attn_weights = torch.softmax(pre_attn @ self.context_vector)
        document_embed = torch.sum(attn_weights * out)
        return document_embed

