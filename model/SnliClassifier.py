import torch
from torch import nn

from model.SentenceEncoder import SentenceEncoder

class MLP(nn.Module):
	def __init__(self, args, embedding_dim, device):
		super(MLP, self).__init__()

		self.num_classes = 3

		if args.skip_connection:
			self.n_dim = 4 * (2 * args.hidden_dim + embedding_dim)
		else:
			self.n_dim = 4 * 2 * args.hidden_dim

		self.encoder = SentenceEncoder(embedding_dim, args.hidden_dim, args.num_layers, args.sent_encoder_dropout_rate, device, args.skip_connection)

		self.layers = nn.Sequential(nn.Linear(self.n_dim, self.num_classes),
									nn.Softmax(dim = 0))

	def forward(self, x1, x2, len1, len2, recover_idx1, recover_idx2):

		# encode sentence batches
		# the _forward_hyperpartisan is exactly what we need
		# just forwards a bunch of sorted sentences
		emb1_sorted, _ = self.encoder._forward_hyperpartisan(x1, len1)
		emb2_sorted, _ = self.encoder._forward_hyperpartisan(x2, len2)

		emb1 = torch.index_select(emb1_sorted, 0, recover_idx1)
		emb2 = torch.index_select(emb2_sorted, 0, recover_idx2)

		# concatenate bacthed embeddings along second dimension
		concatenated_embeddings = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)

		# get logits from classification model
		logits = self.layers(concatenated_embeddings)

		return logits