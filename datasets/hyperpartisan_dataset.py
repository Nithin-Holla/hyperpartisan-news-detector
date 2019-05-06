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

	_pad_token = '<PAD>'
	_unk_token = '<UNK>'

	def __init__(self, filename: str, embedding_vector: Vectors, embedding_dimension: int = 300, use_data: int = 10**6):

		self._hyperpartisan, self._bias, self._title, self._body, self._word_vectors, _vocabulary = self._parse_csv_file(filename, embedding_vector, use_data)

		self._word2id = {
				self._pad_token: 0,
				self._unk_token: 1
			}

		self._id2word = {
				0: self._pad_token,
				1: self._unk_token
			}

		for word in _vocabulary:
			assigned_index = len(self._word2id)
			self._word2id[word] = assigned_index
			self._id2word[assigned_index] = word

		self._data_size = len(self._hyperpartisan)
		self._token_size = len(self._word2id)
		self._embedding = self._create_embedding(embedding_dimension)

	def __getitem__(self, idx):

		title = self._title[idx]
		body = self._body[idx]

		embedded_title = self._embed_sentence(title)
		embedded_body = self._embed_body(body)

		hyperpartisan = torch.LongTensor([self._hyperpartisan[idx]])
		bias = torch.LongTensor([self._bias[idx]])

		title_length = torch.LongTensor([len(title)])
		body_length_in_sent = torch.LongTensor([len(body)])
		body_length_in_tokens = torch.LongTensor([len(sent) for sent in body])

		return hyperpartisan, bias, embedded_body, body_length_in_sent, body_length_in_tokens, embedded_title, title_length

	def __len__(self):
		return self._data_size

	def _parse_csv_file(self, filename: str, embedding_vector: Vectors, use_data: int) \
			-> Tuple[List[bool], List[int], List[str], List[List[str]], Dict[str, torch.Tensor], Set[str]]:
		'''
		Parses the metaphor CSV file and creates the necessary objects for the dataset

		:param str filename: the path to the metaphor CSV dataset file
		:param Vectors embedding_vector: the vector which will be used for word representation
		:return: list of all sentences, list of their labels, dictionary of all the word representations and vocabulary with all the unique words
		'''
		hyperpartisan = []
		bias = []
		title_tokens = []
		body_tokens = []
		word_vectors = {}

		with open(filename, 'r', encoding = "ISO-8859-1") as csv_file:
			next(csv_file)
			csv_reader = csv.reader(csv_file, delimiter='\t')
			data_size = sum(1 for row in csv_reader)
			if use_data < data_size:
				data_size = use_data

		with open(filename, 'r', encoding = "ISO-8859-1") as csv_file:
			next(csv_file)

			csv_reader = csv.reader(csv_file, delimiter='\t')

			for i, row in enumerate(csv_reader):

				hyperpartisan.append(int(row[1]))
				bias.append(int(row[2]))
				title_tokens.append(literal_eval(row[3]))

				for token in title_tokens[-1]:
					if token not in word_vectors.keys():
						word_vectors[token] = embedding_vector[token]

				body = []
				sent = []
				for token in literal_eval(row[5])[1:-1]:
					if token == "</s>":
						body.append(sent)
						sent = []
					elif token == "<s>":
						pass
					else:
						sent.append(token)

					if token not in word_vectors.keys():
						word_vectors[token] = embedding_vector[token]

				body_tokens.append(body)

				if not i % 10000 and i != 0:
					print("{} out of {} loaded".format(i, use_data))

				if i == use_data:
					break

		vocabulary = set(word_vectors.keys())

		return hyperpartisan, bias, title_tokens, body_tokens, word_vectors, vocabulary

	def _create_embedding(self, embedding_dimension: int) -> nn.Embedding:
		'''
		Creates an embedding layer including all word vectors 

		:param str embedding_dimension: the dimensions of the original embedding vector
		:return: the created embedding layer
		'''
		all_embeddings = torch.stack(list(self._word_vectors.values()))
		embeddings_mean = float(all_embeddings.mean())
		embeddings_std = float(all_embeddings.std())

		# Initialize an embedding matrix of (vocab_size, embedding_dimension) shape with
		# random numbers and with a similar distribution as the pretrained embeddings for words in vocab.
		embedding_matrix = torch.FloatTensor(
			self._token_size, embedding_dimension).normal_(embeddings_mean, embeddings_std)

		# Go through the embedding matrix and replace the random vector with a
		# pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
		for i in range(2, self._token_size):
			word = self._id2word[i]
			if word in self._word_vectors:
				embedding_matrix[i] = torch.FloatTensor(
					self._word_vectors[word])

		embeddings = nn.Embedding(
			self._token_size, embedding_dimension, padding_idx=0)

		embeddings.weight = nn.Parameter(embedding_matrix, requires_grad=False)
		return embeddings

	def _embed_sentence(self, sentence: str) -> torch.Tensor:
		'''
		Embeds the given sentence using the initial embedding layer

		:param str sentence: the sentence to be embedded
		:return: a tensor representation of the sentence
		'''
		indexed_sequence = [self._word2id.get(token, 1) for token in sentence]

		result = self._embedding(Variable(torch.LongTensor(indexed_sequence)))
		return result

	def _embed_body(self, body: list) -> list:
		'''
		Gets a list of tensor from the sentences of the body

		:param list body: list of list of tokens
		:return: a list of tensors
		'''

		result = [self._embed_sentence(sent) for sent in body]
		return result