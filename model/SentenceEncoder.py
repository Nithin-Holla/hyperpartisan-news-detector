import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SentenceEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, device):
        super(SentenceEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tasks = ['hyperpartisan', 'metaphor']

        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               num_layers=2, bidirectional=True, batch_first=True)

        self.pre_attn = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                                      nn.Tanh())
        self.context_vector = nn.Parameter(torch.randn((2 * hidden_dim, 1)))
        self.metaphor_fc = nn.Sequential(nn.Linear(2 * hidden_dim, 1),
                                         nn.Sigmoid())
        self.device = device
        self.tasks = ['hyperpartisan', 'metaphor']

    def forward(self, x, len_x, task):
        assert task in self.tasks

        if task == 'hyperpartisan':
            packed_seq = pack_padded_sequence(x, len_x, batch_first=True)
            out, _ = self.encoder(packed_seq)
            pad_packed_states, _ = pad_packed_sequence(out, batch_first=True)
            pre_attn = self.pre_attn(pad_packed_states)
            # Masked attention
            max_len = x.shape[1]
            mask = torch.arange(max_len, device=self.device)[None, :] < len_x[:, None]
            mask = mask.unsqueeze(2)
            dot_product = pre_attn @ self.context_vector
            dot_product[~mask] = float('-inf')
            attn_weights = torch.softmax(dot_product, dim=1)
            attn_weights = attn_weights.expand_as(pad_packed_states)
            sentence_embed = torch.sum(attn_weights * pad_packed_states, dim=1)

            return sentence_embed
        elif task == 'metaphor':
            packed_seq = pack_padded_sequence(x, len_x, batch_first=True)
            out, _ = self.encoder(packed_seq)
            pad_packed_states, _ = pad_packed_sequence(out, batch_first=True)
            pred = self.metaphor_fc(pad_packed_states)
            return pred
