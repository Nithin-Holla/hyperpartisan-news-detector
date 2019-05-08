import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class WordEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_vectors, device):
        super(WordEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False  # Fixed embedding or fine-tuned?
        self.encoder = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pre_attn = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                                      nn.Tanh())
        self.context_vector = nn.Parameter(torch.ones((2 * hidden_dim, 1)))
        self.metaphor_fc = nn.Sequential(nn.Linear(2 * hidden_dim, 1),
                                         nn.Sigmoid())
        self.device = device
        self.tasks = ['hyperpartisan', 'metaphor']

    def forward(self, x, len_x, task):
        assert task in self.tasks
        if task == 'hyperpartisan':
            embed = self.embedding(x)
            out, hidden = self.encoder(embed)  # slice and reorder out later
            pre_attn = self.pre_attn(out)
            # Masked attention
            max_len = x.shape[1]
            mask = torch.arange(max_len, device=self.device)[None, :] < len_x[:, None]
            mask = mask.unsqueeze(2)
            dot_product = pre_attn @ self.context_vector
            dot_product[~mask] = float('-inf')
            attn_weights = torch.softmax(dot_product, dim=1)
            attn_weights = attn_weights.expand_as(out)
            sentence_embed = torch.sum(attn_weights * out, dim=1)
            return sentence_embed
        elif task == 'metaphor':
            embed = self.embedding(x)
            sorted_lengths, sort_indices = torch.sort(len_x, descending=True)
            embed = embed[sort_indices, :, :]
            packed_seq = pack_padded_sequence(embed, sorted_lengths, batch_first=True)
            out, _ = self.encoder(packed_seq)
            pad_packed_states, _ = pad_packed_sequence(out, batch_first=True)
            _, unsorted_indices = torch.sort(sort_indices)
            pad_packed_states = pad_packed_states[unsorted_indices, :, :]
            flipped_states = torch.cat((pad_packed_states[:, :, 0:self.hidden_dim],
                                        torch.flip(pad_packed_states[:, :, self.hidden_dim:], dims=[2])), dim=2)
            pred = self.metaphor_fc(flipped_states)
            return pred

