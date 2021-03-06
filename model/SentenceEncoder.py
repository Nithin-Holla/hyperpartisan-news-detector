import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from enums.training_mode import TrainingMode

import math

class SentenceEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout_rate, device, skip_connection, pre_attn_layer):
        super(SentenceEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               num_layers=num_layers, bidirectional=True, batch_first=True)
        if skip_connection:
            pre_attn_input_dim = 2 * hidden_dim + embedding_dim
        else:
            pre_attn_input_dim = 2 * hidden_dim

        if pre_attn_layer:
            self.pre_attn = nn.Sequential(nn.Dropout(p=dropout_rate),
                                          nn.Linear(pre_attn_input_dim, 2 * hidden_dim),
                                          nn.Tanh())
        else:
            self.pre_attn = nn.Sequential(nn.Dropout(p=dropout_rate))
            
        self.context_vector = nn.Parameter(torch.randn((2 * hidden_dim, 1)))
        stdv = 1. / math.sqrt(self.context_vector.size(0))
        self.context_vector.data.normal_(mean=0, std=stdv)

        self.metaphor_fc = nn.Sequential(nn.Linear(2 * hidden_dim, 1),
                                         nn.Sigmoid())
        self.device = device
        self.skip_connection = skip_connection

    def forward(self, x, len_x, task: TrainingMode):
        if task == TrainingMode.Hyperpartisan:
            sentence_embed, sentence_attn = self._forward_hyperpartisan(x, len_x)
            return sentence_embed, sentence_attn
        elif task == TrainingMode.Metaphor:
            prediction = self._forward_metaphor(x, len_x)
            return prediction
        else:
            raise Exception('Invalid task argument')

    def _forward_hyperpartisan(self, x, len_x):
        packed_seq = pack_padded_sequence(x, len_x, batch_first=True)
        out, _ = self.encoder(packed_seq)
        pad_packed_states, _ = pad_packed_sequence(out, batch_first=True)
        if self.skip_connection:
            pad_packed_states = torch.cat([pad_packed_states, x[:, :, :self.embedding_dim]], dim=2)
        pre_attn = self.pre_attn(pad_packed_states)

        # Masked attention
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=self.device)[None, :] < len_x[:, None]
        mask = mask.unsqueeze(2)
        dot_product = pre_attn @ self.context_vector
        dot_product[~mask] = float('-inf')
        attn_weights = torch.softmax(dot_product, dim=1)
        sentence_embed = torch.sum(attn_weights.expand_as(pad_packed_states) * pad_packed_states, dim=1)

        return sentence_embed, attn_weights

    def _forward_metaphor(self, x, len_x):
        packed_seq = pack_padded_sequence(x, len_x, batch_first=True)
        out, _ = self.encoder(packed_seq)
        pad_packed_states, _ = pad_packed_sequence(out, batch_first=True)
        prediction = self.metaphor_fc(pad_packed_states)

        return prediction
