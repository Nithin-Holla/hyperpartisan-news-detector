import torch
from torch import nn

from model.DocumentEncoder import DocumentEncoder
from model.SentenceEncoder import SentenceEncoder

from enums.training_mode import TrainingMode


class JointModel(nn.Module):

    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            num_layers,
            sent_encoder_dropout_rate,
            doc_encoder_dropout_rate,
            output_dropout_rate,
            device):

        super(JointModel, self).__init__()
        self.sentence_encoder = SentenceEncoder(
            embedding_dim, hidden_dim, num_layers, sent_encoder_dropout_rate, device)
        self.document_encoder = DocumentEncoder(
            2 * hidden_dim, hidden_dim, doc_encoder_dropout_rate, device)
        self.hyperpartisan_fc = nn.Sequential(nn.Dropout(p = output_dropout_rate),
                                              nn.Linear(2 * hidden_dim + 18, 1),
                                              nn.Sigmoid())

        self.device = device

    def forward(
            self,
            x,
            extra_args,
            task: TrainingMode,
            return_attention=False):
            
        if task == TrainingMode.Hyperpartisan:
            recover_idx, num_sent_per_document, sent_lengths, doc_features = extra_args
            out, word_attn, sent_attn = self._forward_hyperpartisan(x, recover_idx, num_sent_per_document, sent_lengths, doc_features)
        elif task == TrainingMode.Metaphor:
            assert return_attention is False, 'Attention is used only in hyperpartisan mode'
            len_x = extra_args
            out = self._forward_metaphor(x, len_x)
        else:
            raise Exception('Invalid task argument')

        if not return_attention:
            return out
        else:
            return out, word_attn, sent_attn

    def _forward_hyperpartisan(self, x, recover_idx, num_sent_per_document, sent_lengths, doc_features):
        # extra_args argument contains the recover_idx to unsort sentences
        # and a list of the number of sentences per article to batch them
        batch_size = len(num_sent_per_document)

        sorted_sent_embeddings, sorted_word_attn = self.sentence_encoder(
            x, sent_lengths, TrainingMode.Hyperpartisan)

        # unsort
        sent_embeddings_2d = torch.index_select(
            sorted_sent_embeddings, dim=0, index=recover_idx)

        word_attn = torch.index_select(sorted_word_attn, dim=0, index=recover_idx).squeeze(2)

        sorted_idx_sent = torch.argsort(
            num_sent_per_document, descending=True)

        sorted_num_sent_per_document = torch.index_select(
            num_sent_per_document, dim=0, index=sorted_idx_sent)

        max_num_sent = sorted_num_sent_per_document[0]

        # create new 3d tensor (already padded across dim=1)
        sent_embeddings_3d = torch.zeros(
            batch_size, max_num_sent.item(), sent_embeddings_2d.shape[-1]).to(self.device)

        # fill the 3d tensor
        processed_sent = 0
        for i, num_sent in enumerate(num_sent_per_document):

            sent_embeddings_3d[i, :num_sent.item(
            ), :] = sent_embeddings_2d[processed_sent: processed_sent + num_sent.item(), :]
            processed_sent += num_sent.item()

        sorted_sent_embeddings_3d = torch.index_select(
            sent_embeddings_3d, dim=0, index=sorted_idx_sent)

        # get document embeddings
        sorted_doc_embedding, sorted_sent_attn = self.document_encoder(
            sorted_sent_embeddings_3d, sorted_num_sent_per_document)

        recover_idx_sent = torch.argsort(sorted_idx_sent, descending=False)

        doc_embedding = torch.index_select(
            sorted_doc_embedding, dim=0, index=recover_idx_sent)

        doc_embedding = torch.cat([doc_embedding, doc_features], dim = 1)

        sent_attn = torch.index_select(sorted_sent_attn, dim=0, index=recover_idx_sent).squeeze(2)

        out = self.hyperpartisan_fc(doc_embedding).view(-1)

        return out, word_attn, sent_attn

    def _forward_metaphor(self, x, len_x):
        out = self.sentence_encoder(x, len_x, TrainingMode.Metaphor)
        return out