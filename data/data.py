import pandas as pd 
import numpy as np
from ast import literal_eval

class Dataset(object):
	def __init__(self, args):

		s = ""
		if args.lowercase:
			s = "_lower"

		self.use_title = False

		self.batch_size = args.batch_size

		# take only part of the sets (for debugging)
		if args.percentage_of_train_data == 100:
			nrows_train = None
		elif args.percentage_of_train_data == 0:
			nrows_train = 1
		else:
			nrows_train = int(args.percentage_of_train_data * len(pd.read_csv(args.data_path + "train_invidx" + s + ".txt", sep = "\t", usecols = [0])) / 100)

		if args.percentage_of_valid_data == 100:
			nrows_valid = None
		elif args.percentage_of_valid_data == 0:
			nrows_valid = 1
		else:
			nrows_valid = int(args.percentage_of_valid_data * len(pd.read_csv(args.data_path + "valid_invidx" + s + ".txt", sep = "\t", usecols = [0])) / 100)

		if args.percentage_of_test_data == 100:
			nrows_test = None
		else:
			nrows_test = int(args.percentage_of_test_data * len(pd.read_csv(args.data_path + "test_invidx" + s + ".txt", sep = "\t", usecols = [0])) / 100)

		if self.use_title:
			usecols = [0, 1, 5, 6, 7, 8]
		else:
			# hyperpartisan, bias, n_sent_body, body_idx, n_tokens_body
			usecols = [0, 1, 3, 6, 8]

		# load processed Hyperpartisan datasets
		converters = {"title_tokens": literal_eval, "body_tokens": literal_eval, "body_idx": literal_eval, "title_idx": literal_eval}
		dtype = {"hyperpartisan": int, "bias": int, "n_tokens_title": int, "n_tokens_body": int, "n_sent_body": int}
		parse_options = {"sep": "\t", "index_col": 0, "nrows": nrows_train, "dtype": dtype, "converters": converters,
						 "encoding": "ISO-8859-1", "usecols": usecols}

		self.data = {
			"train": pd.read_csv(args.data_path + "train_invidx" + s + ".txt", **parse_options),
			"valid": pd.read_csv(args.data_path + "valid_invidx" + s + ".txt", **parse_options),
			"test": pd.read_csv(args.data_path + "test_invidx" + s + ".txt", **parse_options)
			}

		# dictionary for SNLI dataset sizes
		self.size = {"train": len(self.data["train"]), "valid": len(self.data["valid"]), "test": len(self.data["test"])}

		# load full vocab
		with open(args.data_path + "vocabulary" + s + ".txt", "r") as f:
			self.n_vocab_full = len(f.read().splitlines())

		# load created embeddings (np array)
		self.embeddings = pd.read_csv(args.data_path  + "embeddings" + s + ".csv", sep = " ", index_col = 0)
		self.n_vocab, self.n_dim = self.embeddings.shape
		self.vocab = set(self.embeddings.index)
		self.embeddings = self.embeddings.values

		# initialize index for batch scheduler
		self.index = 0

		print("   Finished loading Hyperpartisan datatset with train: {}, valid: {}, test: {}".format(self.size["train"], self.size["valid"], self.size["test"]))

		print("   Finished loading {}-d embeddings for {} tokens out of {}".format(self.n_dim, self.n_vocab, self.n_vocab_full))


	def get_embeddings(self, column):

		n_tokens = "n_tokens_{}".format(column)
		tokens = "{}_tokens".format(column)

		lens = self.batch_data.loc[:, n_tokens].values
		max_len = lens.max()

		batch_emb = np.zeros([max_len, self.batch_size, self.n_dim])

		emb = batch_data[tokens].apply(lambda x: self.embeddings[x, :])

		for i in range(self.batch_size):
			batch_emb[:self.batch_data[n_tokens].iloc[i], i, :] = emb.iloc[i]

		return batch_emb


	def split_body_in_sent(self, full_emb):




	def get_next_batch(self, data_set):

		# get batch from datafame
		self.batch_data = self.data[data_set][self.index: self.index + self.batch_size]

		# increase index (for the next batch)
		self.index += self.batch_size

		# target labels in numpy arrays (not one-hot encoded)
		batch_hyperpartisan = self.batch_data.hyperpartisan.values
		batch_bias = self.batch_data.bias.values

		# lengths
		if self.use_title:
			batch_title_emb = self.get_embeddings("title")
			title_tuple = (batch_title_emb, title_lens)
		else:
			title_tuple = None

		batch_body_emb = self.get_embeddings("body")

		body_tuple = (batch_body_emb, body_lens)

		return batch_hyperpartisan, batch_bias, title_tuple, body_tuple

	def reset_train_set(self):
		self.data["train"] = self.data["train"].iloc[np.random.permutation(len(self.data["train"])), :]
		self.index = 0
