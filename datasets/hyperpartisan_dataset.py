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

import h5py

class HyperpartisanDataset(data.Dataset):

	def __init__(self, filename: str, word_vector: Vectors, elmo_embeddings: int = 1024):

		self._hyperpartisan, self._title, self._body, self._word_vectors = self._parse_csv_file(
			filename, word_vector)
		
		self.elmo_embeddings = elmo_embeddings
		self.article_indexes = {}
		start_index = 0

		for index, current_body in enumerate(self._body):
			end_index = start_index + len(current_body) + 1
			self.article_indexes[index] = (start_index, end_index)
			start_index = end_index

		self.elmo_filename = self._assert_elmo_vectors_file(filename, self._title, self._body)

		self._data_size = len(self._hyperpartisan)
		self.word_vector = word_vector

	def __getitem__(self, idx):
		# print(idx)
		start_index, end_index = self.article_indexes[idx]
		# print(f'start - {start_index}; end - {end_index}')

		elmo_embedding_file = h5py.File(self.elmo_filename, 'r')
		
		title = self._title[idx]
		# print(title)
		body = self._body[idx]

		title_indexed_seq = torch.stack([self.word_vector[token] for token in title])

		body_indexed_seq = []
		for sentence in body:
			word_tensors = torch.stack([self.word_vector[token] for token in sentence])
			body_indexed_seq.append(word_tensors)

		indexed_seq = [title_indexed_seq] + body_indexed_seq

		result_sequences = []
		for index in range(start_index, end_index):
			glove_embeddings = indexed_seq[index - start_index]
			
			sentence_elmo_embeddings = elmo_embedding_file[str(index)]
			elmo_embeddings = torch.cat([torch.Tensor(sentence_elmo_embeddings[0]), torch.Tensor(
				sentence_elmo_embeddings[1]), torch.Tensor(sentence_elmo_embeddings[2])], dim=1)

			# # elmo: [ n_words x (1024*3) ]; [ n_words x 300 ] => [ n_words x 1324 ]
			# assert list(elmo_embeddings.size()) == [sentence_length, self.elmo_dimensions * len(sentence_elmo_embeddings)]
			# assert list(indexed_sequence.size()) == [sentence_length, self.word_vector.dim]
			# print(f'elmo - {elmo_embeddings.shape}')
			# print(f'glove - {glove_embeddings.shape}')

			combined_embeddings = torch.cat([elmo_embeddings, glove_embeddings], dim=1)
			result_sequences.append(combined_embeddings)


		elmo_embedding_file.close()

		hyperpartisan = self._hyperpartisan[idx]

		title_length = len(title)
		body_length_in_sent = len(body)
		body_length_in_tokens = [title_length] + [len(sent) for sent in body]

		assert len(body_length_in_tokens) == len(result_sequences)

		return result_sequences, hyperpartisan, body_length_in_tokens, body_length_in_sent

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
		title_tokens = []
		body_tokens = []
		word_vectors = {}

		with open(filename, 'r', encoding="utf-8") as csv_file:
			next(csv_file)

			csv_reader = csv.reader(csv_file, delimiter='\t')

			for _, row in enumerate(csv_reader):

				hyperpartisan.append(int(row[1]))
				current_title_tokens = literal_eval(row[2])
				title_tokens.append(current_title_tokens)

				for token in title_tokens[-1]:
					if token not in word_vectors.keys():
						word_vectors[token] = embedding_vector[token]

				current_body_tokens = literal_eval(row[4])
				body_tokens.append(current_body_tokens)

		return hyperpartisan, title_tokens, body_tokens, word_vectors

	def _assert_elmo_vectors_file(self, csv_filename, title_tokens, body_tokens):
		dirname = os.path.dirname(csv_filename)
		filename_without_ext = os.path.splitext(
			os.path.basename(csv_filename))[0]
		elmo_filename = os.path.join(dirname, f'{filename_without_ext}.hdf5')
		if not os.path.isfile(elmo_filename):
			print("caching elmo vectors")
			sentences_filename = os.path.join(
				dirname, f'{filename_without_ext}_elmo.txt')
			with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
				for index, article_tokens in enumerate(body_tokens):
					article_title_tokens = title_tokens[index]
					title_text = ' '.join(article_title_tokens)
					sentences_file.write(f'{title_text}\n')

					for sentence_tokens in article_tokens:
						sentence_text = ' '.join(sentence_tokens)
						sentences_file.write(f'{sentence_text}\n')

			raise Exception(
				'Please save the sentences file to elmo file using \'allennlp elmo\' command')

		return elmo_filename
