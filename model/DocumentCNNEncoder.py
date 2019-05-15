import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class DocumentCNNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, device):
        super(DocumentCNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        Ks = [2,3,4,5,6]
        
        self.convs1 = nn.ModuleList([nn.Conv2d(1, hidden_dim, (K, 256)) for K in Ks])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim, momentum=0.7) for _ in Ks])
        self.device = device

    def forward(self, x, len_x):
        x = [torch.nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        
        x = [batch_norm(x[i]) for i, batch_norm in enumerate(self.batch_norms)]  # [(N, Co), ...]*len(Ks)

        x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        attn_weights = torch.zeros(x.shape[0], device=self.device)
        return x, attn_weights

