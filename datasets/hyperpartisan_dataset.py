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

	def __init__(
			self,
			filename: str,
			glove_vectors: Vectors,
			lowercase_sentences: bool = False,
            articles_max_length: int = None):

		self.glove_vectors = glove_vectors
		self.lowercase_sentences = lowercase_sentences
		self.articles_max_length = articles_max_length

		self._labels, self._title_tokens, self._body_tokens, self._extra_feat = self._parse_csv_file(
			filename)

		self.article_indexes = {}
		start_index = 0

		for index, current_body in enumerate(self._body_tokens):
			end_index = start_index + len(current_body) + 1
			self.article_indexes[index] = (start_index, end_index)
			start_index = end_index

		self.elmo_filename = self._assert_elmo_vectors_file(
			filename, self._title_tokens, self._body_tokens)

		self._data_size = len(self._labels)

	def __getitem__(self, idx):

		start_index, end_index = self.article_indexes[idx]

		elmo_embedding_file = h5py.File(self.elmo_filename, 'r')

		extra_feat = self._extra_feat[idx]
		title_tokens = self._title_tokens[idx]
		body_tokens = self._body_tokens[idx]

		if self.lowercase_sentences:
			title_tokens = [token.lower() for token in title_tokens]
			body_tokens = [token.lower() for token in sentence_tokens for sentence_tokens in title_tokens]

		title_glove_embeddings = torch.stack(
			[self.glove_vectors[x] if x in self.glove_vectors.stoi else self.glove_vectors[x.lower()] for x in title_tokens])

		body_glove_embeddings = []
		current_tokens_amount = len(title_tokens)
		for sentence_tokens in body_tokens:
			if self.articles_max_length and current_tokens_amount >= self.articles_max_length:
				break
			elif self.articles_max_length and current_tokens_amount + len(sentence_tokens) > self.articles_max_length:
				sentence_tokens = sentence_tokens[0:(self.articles_max_length - current_tokens_amount)]

			current_tokens_amount += len(sentence_tokens)
			sentence_glove_embeddings = torch.stack(
				[self.glove_vectors[x] if x in self.glove_vectors.stoi else self.glove_vectors[x.lower()] for x in sentence_tokens])
			body_glove_embeddings.append(sentence_glove_embeddings)

		glove_embeddings = [title_glove_embeddings] + body_glove_embeddings

		result_embeddings = []
		current_tokens_amount = 0
		for index in range(start_index, end_index):
			if self.articles_max_length and current_tokens_amount >= self.articles_max_length:
				break

			sentence_glove_embeddings = glove_embeddings[index - start_index]
			sentence_elmo_embeddings = torch.Tensor(elmo_embedding_file[str(index)])
			
			if self.articles_max_length and current_tokens_amount + len(sentence_elmo_embeddings) > self.articles_max_length:
				sentence_elmo_embeddings = sentence_elmo_embeddings[0:(self.articles_max_length - current_tokens_amount)]

			current_tokens_amount += len(sentence_elmo_embeddings)

			# elmo: [ n_words x (1024) ]; glove: [ n_words x 300 ] => combined: [ n_words x 1324 ]
			# combined_embeddings = torch.cat(
			# 	[sentence_elmo_embeddings, sentence_glove_embeddings], dim=1)
			result_embeddings.append(sentence_elmo_embeddings)

		elmo_embedding_file.close()

		is_hyperpartisan = self._labels[idx]

		body_sentences_amount = len(result_embeddings)
		body_tokens_per_sentence = [len(sentence_embeddings) for sentence_embeddings in result_embeddings]

		assert len(body_tokens_per_sentence) == body_sentences_amount

		# print(len(result_embeddings), len(is_hyperpartisan), len(body_tokens_per_sentence), len(body_sentences_amount), len(extra_feat))

		return result_embeddings, is_hyperpartisan, body_tokens_per_sentence, body_sentences_amount, extra_feat

	def __len__(self):
		return self._data_size

	def _parse_csv_file(self, filename: str) \
			-> Tuple[List[bool], List[str], List[List[str]]]:
		'''
		Parses the metaphor CSV file and creates the necessary objects for the dataset

		:param str filename: the path to the metaphor CSV dataset file
		:return: list of all sentences, list of their labels, dictionary of all the word representations and vocabulary with all the unique words
		'''
		labels = []
		title_tokens = []
		body_tokens = []
		ids = []
		extra_feat = []

		month_dict = {}
		for month in range(12):
			z = np.zeros(12)
			z[month] = 1
			z = list(z)
			month_dict[month] = z

		with open(filename, 'r', encoding="utf-8") as csv_file:
			next(csv_file)

			csv_reader = csv.reader(csv_file, delimiter='\t')

			for _, row in enumerate(csv_reader):

				ids.append(int(row[0]))
				
				date = row[1]
				try:
					y = int(date[:4])
					m = int(date[5:7])

					if y == 2018:
						xtr = [0, 0, 0, 1]
					elif y == 2017:
						xtr = [0, 0, 1, 0]
					elif y == 2016:
						xtr = [0, 1, 0, 0]
					else:
						xtr = [0, 0, 0, 0]

					xtr += month_dict[m]
				except:
					xtr = [0]*16

				xtr.append(float(row[-2]))
				xtr.append(float(row[-1]))

				extra_feat.append(xtr)

				labels.append(int(row[4] == "True"))
				current_title_tokens = literal_eval(row[2])

				title_tokens.append(current_title_tokens)

				current_body_tokens = literal_eval(row[3])

				body_tokens.append(current_body_tokens)

				assert len(xtr) == 18

		return labels, title_tokens, body_tokens, extra_feat

	def _assert_elmo_vectors_file(self, csv_filename, title_tokens, body_tokens):
		dirname = os.path.dirname(csv_filename)
		filename_without_ext = os.path.splitext(
			os.path.basename(csv_filename))[0]

		file_suffix = self._create_elmo_file_suffix()
		elmo_filename = os.path.join(
			dirname, f'{filename_without_ext}{file_suffix}.hdf5')

		if not os.path.isfile(elmo_filename):
			print("Caching elmo vectors...")
			sentences_filename = os.path.join(
				dirname, f'{filename_without_ext}{file_suffix}.txt')
			with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
				for index, article_tokens in enumerate(body_tokens):
					article_title_tokens = title_tokens[index]
					title_text = ' '.join(article_title_tokens)
					
					if self.lowercase_sentences:
						title_text = title_text.lower()

					sentences_file.write(f'{title_text}\n')

					for sentence_tokens in article_tokens:
						sentence_text = ' '.join(sentence_tokens)
						if self.lowercase_sentences:
							sentence_text = sentence_text.lower()

						sentences_file.write(f'{sentence_text}\n')

			raise Exception(
				f'Please save the sentences file to the file {filename_without_ext}{file_suffix}.hdf5 using \'allennlp elmo\' command')

		return elmo_filename

	def _create_elmo_file_suffix(self):
		'''
		Creates a file suffix which includes all current configuration options
		'''

		file_suffix = '_elmo'

		if self.lowercase_sentences:
			file_suffix = f'_lowercase{file_suffix}'

		return file_suffix
