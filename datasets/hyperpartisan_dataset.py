import torchtext
import os
import numpy as np

from torchtext.vocab import Vectors

from torch.autograd import Variable
import torch.utils.data as data
import torch
import torch.nn as nn
from ast import literal_eval

import csv
from typing import List, Tuple, Dict, Set

class HyperpartisanDataset(data.Dataset):

	def __init__(self, filename: str, word_vector: Vectors):

		self._hyperpartisan, self._title, self._body, self._word_vectors = self._parse_csv_file(filename, word_vector)

		self._data_size = len(self._hyperpartisan)
		self.word_vector = word_vector

	def __getitem__(self, idx):

		title = self._title[idx]
		body = self._body[idx]

		title_indexed_seq = [self.word_vector.stoi.get(token, 0) for token in title]
		body_indexed_seq = [[self.word_vector.stoi.get(token, 0) for token in sent] for sent in body]

		hyperpartisan = self._hyperpartisan[idx]
		# bias = self._bias[idx]

		title_length = len(title)
		body_length_in_sent = len(body)
		body_length_in_tokens = [len(sent) for sent in body]

		return hyperpartisan, body_indexed_seq, body_length_in_sent, body_length_in_tokens, title_indexed_seq, title_length

	def __len__(self):
		return self._data_size

	def _parse_csv_file(self, filename: str, embedding_vector: Vectors) \
			-> Tuple[List[bool], List[str], List[List[str]], Dict[str, torch.Tensor], Set[str]]:
		'''
		Parses the metaphor CSV file and creates the necessary objects for the dataset

		:param str filename: the path to the metaphor CSV dataset file
		:param Vectors embedding_vector: the vector which will be used for word representation
		:return: list of all sentences, list of their labels, dictionary of all the word representations and vocabulary with all the unique words
		'''
		hyperpartisan = []
		# bias = []
		title_tokens = []
		body_tokens = []
		word_vectors = {}

		# with open(filename, 'r', encoding = "ISO-8859-1") as csv_file:
		# 	next(csv_file)
		# 	csv_reader = csv.reader(csv_file, delimiter='\t')
		# 	data_size = sum(1 for row in csv_reader)
		# 	if use_data < data_size:
		# 		data_size = use_data

		with open(filename, 'r', encoding = "utf-8") as csv_file:
			next(csv_file)

			csv_reader = csv.reader(csv_file, delimiter='\t')

			for i, row in enumerate(csv_reader):

				hyperpartisan.append(int(row[1]))
				# bias.append(int(row[2]))
				# title_tokens.append([token.lower() for token in literal_eval(row[3])])
				title_tokens.append(literal_eval(row[2]))

				for token in title_tokens[-1]:
					if token not in word_vectors.keys():
						word_vectors[token] = embedding_vector[token]

				# body = []
				# sent = []
				# for token in literal_eval(row[5])[1:-1]:
				# 	if token == "</s>":
				# 		body.append(sent)
				# 		sent = []
				# 	elif token == "<s>":
				# 		pass
				# 	else:
				# 		sent.append(token.lower())

				# 	if token not in word_vectors.keys():
				# 		word_vectors[token] = embedding_vector[token]

				# body_tokens.append(body)

				body_tokens.append(literal_eval(row[4]))

				# if not i % 10000 and i != 0:
				# 	print("{} out of {} loaded".format(i, use_data))

				# if i == use_data:
				# 	break

		return hyperpartisan, title_tokens, body_tokens, word_vectors