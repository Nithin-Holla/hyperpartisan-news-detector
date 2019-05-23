import os
import numpy as np
import pandas as pd

from torchtext.vocab import Vectors
import torch.utils.data as data
import torch

from ast import literal_eval
import h5py

class SnliDataset(data.Dataset):
	def __init__(self, filename: str, glove_vectors: Vectors, use_data: int):

		self.glove_vectors = glove_vectors

		self._sentences1, self._sentences2, self._labels, self._ntokens1, self._ntokens2 = self._parse_csv_file(filename, use_data)

		self.elmo_filename1 = filename.split(".")[0] + "_1_elmo.txt"
		self.elmo_filename2 = filename.split(".")[0] + "_2_elmo.txt"

		self._data_size = len(self._labels)

	def __getitem__(self, idx):

		sentence1 = self._sentences1[idx]
		sentence2 = self._sentences2[idx]
		length1 = self._ntokens1[idx]
		length2 = self._ntokens2[idx]
		label = self._labels[idx]

		# get the GloVe embedding. If it's missing for this word - try lowercasing it
		words_embeddings1 = [self.glove_vectors[x] if x in self.glove_vectors.stoi else self.glove_vectors[x.lower()] for x in sentence1]
		words_embeddings2 = [self.glove_vectors[x] if x in self.glove_vectors.stoi else self.glove_vectors[x.lower()] for x in sentence2]


		glove_embeddings1 = torch.stack(words_embeddings1)
		elmo_embedding_file1 = h5py.File(self.elmo_filename1, 'r')
		elmo_embeddings1 = torch.Tensor(elmo_embedding_file1[str(idx)])

		glove_embeddings2 = torch.stack(words_embeddings2)
		elmo_embedding_file2 = h5py.File(self.elmo_filename2, 'r')
		elmo_embeddings2 = torch.Tensor(elmo_embedding_file2[str(idx)])

		# elmo: [ n_words x (1024) ]; glove: [ n_words x 300 ] => combined: [ n_words x 1324 ]
		combined_embeddings1 = torch.cat([elmo_embeddings1, glove_embeddings1], dim=1)
		combined_embeddings2 = torch.cat([elmo_embeddings2, glove_embeddings2], dim=1)

		elmo_embedding_file1.close()
		elmo_embedding_file2.close()

		return combined_embeddings1, combined_embeddings2, label, length1, length2

	def __len__(self):
		return self._data_size

	def _parse_csv_file(self, filename):

		dtype = {"label": int, "ntokens1": int, "ntokens2": int}
		converters = {"sentence1": literal_eval, "sentence2": literal_eval}
		df = pd.read_csv(filename, index_col = 0, sep = "\t", dtype = dtype, converters = converters, nrows = use_data)

		return df.sentence1.tolist(), df.sentence2.tolist(), df.label.tolist(), df.ntokens1.tolist(), df.ntokens2.tolist()