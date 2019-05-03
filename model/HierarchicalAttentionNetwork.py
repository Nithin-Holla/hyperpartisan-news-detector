import torch
from torch import nn

from model.SentenceEncoder import SentenceEncoder
from model.WordEncoder import WordEncoder


class HierarchicalAttentionNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_vectors):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.word_encoder = WordEncoder(vocab_size, embedding_dim, hidden_dim, pretrained_vectors)
        self.sentence_encoder = SentenceEncoder(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, doc):
        sent_embeddings = []
        for sent in doc:
            sent_embed = self.word_encoder(sent)
            sent_embeddings.append(sent_embed)
        sent_embeddings = torch.stack(sent_embeddings)
        doc_embedding = self.sentence_encoder(sent_embeddings)
        out = self.fc(doc_embedding)
        return out