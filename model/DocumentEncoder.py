import torch
from torch import nn


class DocumentEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, device):
        super(DocumentEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pre_attn = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                                      nn.Tanh())
        self.context_vector = nn.Parameter(torch.ones((2 * hidden_dim, 1)))
        self.device = device

    def forward(self, x, len_x):
        out, hidden = self.encoder(x)   # slice and reorder out later
        pre_attn = self.pre_attn(out)
        # Masked attention
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=self.device)[None, :] < len_x[:, None]
        mask = mask.unsqueeze(2)
        dot_product = pre_attn @ self.context_vector
        dot_product[~mask] = float('-inf')
        attn_weights = torch.softmax(dot_product, dim=1)
        attn_weights = attn_weights.expand_as(out)
        document_embed = torch.sum(attn_weights * out, dim=1)
        return document_embed

