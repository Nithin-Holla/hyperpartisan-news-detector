import torch
from torch import nn

from model.SentenceEncoder import SentenceEncoder
from model.WordEncoder import WordEncoder


class JointModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, hyp_n_classes, pretrained_vectors):
        super(JointModel, self).__init__()
        self.word_encoder = WordEncoder(vocab_size, embedding_dim, hidden_dim, pretrained_vectors)
        self.sentence_encoder = SentenceEncoder(hidden_dim, hidden_dim)
        self.metaphor_fc = nn.Linear(hidden_dim, 1)
        self.hyperpartisan_fc = nn.Linear(hidden_dim, hyp_n_classes)
        self.tasks = ['hyperpartisan', 'metaphor']

    def forward(self, x, task):
        assert task in self.tasks
        if task == 'hyperpartisan':
            sent_embeddings = []
            for sent in x:
                sent_embed = self.word_encoder(sent)
                sent_embeddings.append(sent_embed)
            sent_embeddings = torch.stack(sent_embeddings)
            doc_embedding = self.sentence_encoder(sent_embeddings)
            out = self.hyperpartisan_fc(doc_embedding)
            return out
        elif task == 'metaphor':
            sent_embed = self.word_encoder(x)
            out = self.metaphor_fc(sent_embed)
            return out