import torch
from torch import nn

from model.SentenceEncoder import SentenceEncoder
from model.WordEncoder import WordEncoder


class JointModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, device):
        super(JointModel, self).__init__()
        self.word_encoder = WordEncoder(vocab_size, embedding_dim, hidden_dim, device)
        self.sentence_encoder = SentenceEncoder(2 * hidden_dim, hidden_dim, device)
        self.hyperpartisan_fc = nn.Sequential(nn.Linear(2 * hidden_dim, 1),
                                              nn.Sigmoid())
        self.tasks = ['hyperpartisan', 'metaphor']
        self.device = device

    def forward(self, x, extra_args, task):
        assert task in self.tasks
        if task == 'hyperpartisan':

            # extra_args argument contains the recover_idx to unsort sentences
            # and a list of the number of sentences per article to batch them
            recover_idx, num_sent_per_document, sent_lengths = extra_args
            batch_size = len(num_sent_per_document)

            sorted_sent_embeddings = self.word_encoder(x, sent_lengths, task)

            # unsort
            sent_embeddings_2d = torch.index_select(sorted_sent_embeddings, dim = 0, index = recover_idx)

            sorted_idx_sent = torch.argsort(num_sent_per_document, descending = True)
            sorted_num_sent_per_document = torch.index_select(num_sent_per_document, dim = 0, index = sorted_idx_sent)
            max_num_sent = sorted_num_sent_per_document[-1]

            # create new 3d tensor (already padded across dim=1)
            sent_embeddings_3d = torch.zeros(batch_size, max_num_sent.int().item(), sent_embeddings_2d.shape[-1]).to(self.device)

            # fill the 3d tensor
            processed_sent = 0
            for i, num_sent in enumerate(num_sent_per_document):
                sent_embeddings_3d[i, :num_sent, :] = sent_embeddings_2d[processed_sent: processed_sent + num_sent, :]
                processed_sent += num_sent

            sorted_sent_embeddings_3d = torch.index_select(sent_embeddings_3d, dim = 0, index = sorted_idx)

            # get document embeddings
            sorted_doc_embedding = self.sentence_encoder(sorted_sent_embeddings_3d, sorted_num_sent_per_document)

            recover_idx_sent = torch.argsort(sorted_idx_sent, descending = False)

            doc_embedding = torch.index_select(sorted_doc_embedding, dim = 0, index = recover_idx_sent)

            out = self.hyperpartisan_fc(doc_embedding).view(-1)

        elif task == 'metaphor':
            len_x = extra_args
            out = self.word_encoder(x, len_x, task)

        return out
