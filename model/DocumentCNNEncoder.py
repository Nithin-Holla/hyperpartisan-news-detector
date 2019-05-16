import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class DocumentCNNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, device):
        super(DocumentCNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Ks = [2,3,4,5,6]
        #
        # self.convs1 = nn.ModuleList([nn.Conv2d(1, hidden_dim, (K, 256)) for K in Ks])
        # self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim, momentum=0.7) for _ in Ks])

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(),
                                   nn.MaxPool1d(200 - 2))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(),
                                   nn.MaxPool1d(200 - 3))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=4),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(),
                                   nn.MaxPool1d(200 - 4))
        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(),
                                   nn.MaxPool1d(200 - 5))
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=6),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(),
                                   nn.MaxPool1d(200 - 6))

        self.device = device

    def forward(self, x, len_x):
        # x = [torch.nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        #
        # x = [batch_norm(x[i]) for i, batch_norm in enumerate(self.batch_norms)]  # [(N, Co), ...]*len(Ks)
        #
        # x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        #
        # x = torch.cat(x, 1)

        c1 = self.conv1.forward(x)
        c2 = self.conv2.forward(x)
        c3 = self.conv3.forward(x)
        c4 = self.conv4.forward(x)
        c5 = self.conv5.forward(x)

        c = torch.cat((c1, c2, c3, c4, c5), dim=1)
        c = c.squeeze(2)

        attn_weights = torch.zeros(x.shape[0], device=self.device)
        return c, attn_weights

