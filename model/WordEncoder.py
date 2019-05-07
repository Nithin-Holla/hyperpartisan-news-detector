import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class WordEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_vectors):
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
        self.context_vector = nn.Parameter(torch.zeros((2 * hidden_dim, 1)))
        self.metaphor_fc = nn.Sequential(nn.Linear(2 * hidden_dim, 1),
                                         nn.Softmax(dim=1))
        self.tasks = ['hyperpartisan', 'metaphor']

    def forward(self, x, len_x, task):
        assert task in self.tasks
        if task == 'hyperpartisan':
            embed = self.embedding(x)
            out, hidden = self.encoder(embed)  # slice and reorder out later
            pre_attn = self.pre_attn(out)
            attn_weights = torch.softmax(pre_attn @ self.context_vector, dim=1)
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

