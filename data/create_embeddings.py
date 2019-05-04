import pandas as pd 
import numpy as np
import os
from ast import literal_eval
import argparse
from tqdm import tqdm
tqdm.pandas()

def load_data(data_path):
# return train, valid, test dataframes in a dictionary
# columns per dataframe:
	# hyperpartisan: 0 or 1
	# bias: 0: left, 1: left-center, 2: least, 3: right-center, 4: right
	# n_tokens_title: number of tokens in the title
	# n_tokens_body: number of tokens in the body
	# n_sent_body: number of senetences in the body
	# title_tokens: list of title tokens
	# body_tokens: list of body tokens
	# index: unique article id

	df = {}
	sets = ["train", "valid", "test"]

	converters = {"title_tokens": literal_eval, "body_tokens": literal_eval}
	dtype = {"hyperpartisan": int, "bias": int, "n_tokens_title": int, "n_tokens_body": int, "n_sent_body": int}

	for set_name in sets:
		df[set_name] = pd.read_csv(data_path + set_name + ".txt", sep = "\t", index_col = 0, converters = converters, dtype = dtype, encoding = "ISO-8859-1")
		df[set_name].index = df[set_name].index.astype(int, False)

	return df


def create_vocabulary(df):
# returns a set of all the unique tokens found in title and body

	vocab = set()

	for set_name in df.keys():

		print("Vocabulary from {} set".format(set_name))

		l = len(df[set_name])
		rng = list(range(0, l, 1000)) + [l]

		for j, i1, i2 in zip(range(len(rng) - 1), rng[:-1], rng[1:]):

			print("		{}/{}".format(j + 1, len(rng)))

			vocab.update(df[set_name].title_tokens.iloc[i1:i2].sum())
			vocab.update(df[set_name].body_tokens.iloc[i1:i2].sum())

	return vocab


def build_embeddings(vocab, glove_path, glove_size, chunksize):
# return a dataframe of embeddings indexed by the tokens in the vocabulary

	print("Bulding embeddings ...")
	for i, chunk in enumerate(pd.read_csv(glove_path, header = None, sep = "\s", index_col = 0, engine = "python",
	 error_bad_lines = False, warn_bad_lines = False, chunksize = chunksize, nrows = glove_size)):

		print("Chunk", i + 1)

		# check for some special tokens
		chunk_index = set(chunk.index)
		if "<unk>" in chunk_index:
			print("found <unk>")
		if "<s>" in chunk_index:
			print("found <s>")
		if "</s>" in chunk_index:
			print("found </s>")
		if "<pad>" in chunk_index:
			print("found <pad>")
		if "<num>" in chunk_index:
			print("found <num>")

		# get embeddings for the words in vocabulary for this chunk
		if not i:
			embeddings = chunk.loc[chunk.index.isin(vocab), :]
		else:
			embeddings = pd.concat([embeddings, chunk.loc[chunk.index.isin(vocab), :]], axis = 0)

	# add unknown tokken
	unknown_token = pd.DataFrame(np.random.uniform(-0.05, 0.05, [1, embeddings.shape[1]]), index = ["<unk>"], columns = embeddings.columns)
	embeddings = pd.concat([embeddings, unknown_token], axis = 0)

	# sort index alphabetically
	embeddings.sort_index(inplace = True)

	return embeddings


def build_index(df, embeddings):
# returns the dataframe with two extra columns
# title_idx: index of the tokens in the embeddings dataframe (in order of appearance)
# body_idx: << <<

	emb_idx = pd.Series(range(len(embeddings)), index = embeddings.index)
	unk_idx = int(emb_idx.reindex(["<unk>"]).values)

	for set_name in df.keys():

		print("Inverted index for title tokens ...")
		df[set_name]["title_idx"] = df[set_name]["title_tokens"].progress_apply(lambda x: emb_idx.reindex(x, fill_value = unk_idx).astype(int).values.tolist())
		print("Inverted index for body tokens ...")
		df[set_name]["body_idx"] = df[set_name]["body_tokens"].progress_apply(lambda x: emb_idx.reindex(x, fill_value = unk_idx).astype(int).values.tolist())

	return df

##########################################################################
##########################################################################
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = "./../data/")#"C:/Users/ioann/Datasets/Hyperpartisan/"
parser.add_argument('--glove_path', type = str, default = "./../data//glove.840B.300d.txt")#"C:/Users/ioann/Datasets/glove.840B.300d.txt"
parser.add_argument('--glove_size', type = int, default = None)
parser.add_argument('--chunksize', type = int, default = 10**5)
args, unparsed = parser.parse_known_args()


df = load_data(args.data_path)
print("Data loaded succesfully. train[{}], valid[{}], test[{}]".format(len(df["train"]), len(df["valid"]), len(df["test"]) ))

vocab = create_vocabulary(df)
print("Created vocabulary of size {}".format(len(vocab)))

embeddings = build_embeddings(vocab, args.glove_path, args.glove_size, args.chunksize)
print("Found embeddings for {} tokens".format(len(embeddings)))

df = build_index(df, embeddings)
print("Succesfully constracted inverted index for title and body")

for set_name in df.keys():
	df[set_name].to_csv(args.data_path + set_name + "_invidx.txt", sep = "\t")

embeddings.to_csv(args.data_path + "embeddings.csv", sep = " ")

with open(args.data_path + "vocabulary.txt", "w") as f:
	for token in vocab:
		try:
			f.write(token + "\n")
		except UnicodeEncodeError:
			print(token)

print("Data saved.")