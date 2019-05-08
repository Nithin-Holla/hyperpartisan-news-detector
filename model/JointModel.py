import torch
from torch import nn

from model.SentenceEncoder import SentenceEncoder
from model.WordEncoder import WordEncoder


class JointModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, hyp_n_classes, pretrained_vectors, device):
        super(JointModel, self).__init__()
        self.word_encoder = WordEncoder(vocab_size, embedding_dim, hidden_dim, pretrained_vectors)
        self.sentence_encoder = SentenceEncoder(2*hidden_dim, hidden_dim)
        self.hyperpartisan_fc = nn.Linear(2 * hidden_dim, hyp_n_classes)
        self.tasks = ['hyperpartisan', 'metaphor']
        self.device = device

    def forward(self, x, len_x, task):
        assert task in self.tasks
        if task == 'hyperpartisan':

            # len_x argument contains the recover_idx to unsort sentences
            # and a list of the number of sentences per article to batch them
            recover_idx, num_sent_per_document = len_x
            batch_size = len(num_sent_per_document)

            sorted_sent_embeddings = self.word_encoder(x, len_x, task)

            # unsort
            sent_embeddings_2d = torch.index_select(sorted_sent_embeddings, dim = 0, index = recover_idx)

            max_num_sent = torch.max(num_sent_per_document)

            # create new 3d tensor (already padded across dim=1)
            sent_embeddings_3d = torch.zeros(batch_size, max_num_sent, sent_embeddings_2d.shape[-1]).to(self.device)

            # fill the 3d tensor
            processed_sent = 0
            for i, num_sent in enumerate(num_sent_per_document):
                sent_embeddings_3d[i, :num_sent, :] = sent_embeddings_2d[processed_sent: processed_sent + num_sent, :]
                processed_sent += num_sent

            # get document embeddings
            doc_embedding = self.sentence_encoder(sent_embeddings_3d)

            out = self.hyperpartisan_fc(doc_embedding)

        elif task == 'metaphor':
            out = self.word_encoder(x, len_x, task)

        return out
