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
        self.encoder = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
        self.pre_attn = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                                      nn.Tanh())
        self.context_vector = nn.Parameter(torch.zeros((2 * hidden_dim, 1)))

    def forward(self, x, len_x):
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)   # slice and reorder out later
        pre_attn = self.pre_attn(out)
        attn_weights = torch.softmax(pre_attn @ self.context_vector)
        sentence_embed = torch.sum(attn_weights * out)
        return sentence_embed

