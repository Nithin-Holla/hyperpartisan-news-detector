import pandas as pd 
import numpy as np
from nltk import word_tokenize, sent_tokenize
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
from nltk.corpus import stopwords as nltk_stopwords
import re
import argparse
from copy import copy
from datetime import datetime
from tqdm import tqdm

# use progress bar on pandas apply functions
tqdm.pandas()

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

def load_data(data_path, file_name, filter_body, filter_title, usedata):

	# index is the unique article id
	# cols 1, 6 and 7 are published-at, url and labeled-by
	dtype = {"title": str, "body": str, "hyperpartisan": int}
	df = pd.read_csv(data_path + file_name, sep = "\t", index_col = 0, encoding = "utf-8", dtype = dtype, usecols = [0, 2, 3, 4])
	df.index = df.index.astype(int, False)

	# keep only articles with valid title
	if filter_title:
		len_before = len(df)
		df = df.loc[~df.title.isnull(), :]
		chng = (len_before - len(df)) / len_before * 100
		print("Filtered set for title. Size reduced from {} to {} ({:.2f}%)".format(len_before, len(df), chng))

	# keep only articles with valid body
	if filter_body:
		len_before = len(df)
		df = df.loc[~df.body.isnull(), :]	
		chng = (len_before - len(df)) / len_before * 100
		print("Filtered set for body. Size reduced from {} to {} ({:.2f}%)".format(len_before, len(df), chng))

	# drop duplicate articles
	len_before = len(df)
	df = df.drop_duplicates()
	chng = (len_before - len(df)) / len_before * 100
	print("Dropped duplicates. Size reduced from {} to {} ({:.2f}%)".format(len_before, len(df), chng))

	return df


def clean_data(df):

	def clean_text(text):
	    text = text.replace("&amp;", "&")
	    text = text.replace("&gt;", ">")
	    text = text.replace("&lt;", "<")
	    text = text.replace("<p>", " ")
	    text = text.replace("</p>", " ")
	    text = text.replace(" _", " ")
	    text = text.replace("–", "-")
	    text = text.replace("”", "\"")
	    text = text.replace("“", "\"")
	    text = text.replace("’", "'")
		return text

	# clean text from xml special characters and other inconsistancies
	df["title"] = df.title.apply(clean_text)
	df["body"] = df.body.apply(clean_text)

	return df


def process_data(df, lowercase):

	if lowercase:
		df["title"] = df.title.str.lower
		df["body"] = df.body.str.lower

	# title into word tokens
	print("Title to word tokens ...")
	df["title_tokens"] = df.title.progress_apply(word_tokenize)

	# body into sent tokens
	print("Body to sentence tokens ...")
	df["body_sent"] = df.body.progress_apply(sent_tokenize)
	df["n_sent_body"] = df.body_sent.apply(len)

	# body into word tokens
	print("Body to word tokens ...")
	df["body_tokens"] = df.body_sent.progress_apply(lambda x: [word_tokenize(i) for i in x])

	# delete helper columns
	del df["body_sent"]
	del df["body"]
	del df["title"]

	return df


def filter_data(df):

	# gather stopwords from nltk and gensim
	stopwords = set(nltk_stopwords.words("english"))
	for w in gensim_stopwords:
		stopwords.add(w)
	stopwords.add("'s")
	stopwords.add("?s")
	stopwords.add("n't")
	stopwords.add("n?t")
	for w in copy(stopwords):
		stopwords.add(w.capitalize())
	stopwords.add("...")
	stopwords.add(".....")

	punctuation = '.,:;?!"\'\'``+={}[]()#~$--@'

	for p in list(punctuation):
		stopwords.add(p)

	stopwords.add("--")
	stopwords.add("''")
	stopwords.add("``")

	# print(stopwords)

	re_num_simple = re.compile('^-?[0-9.,]+([eE^][0-9]+)?(th)?$')

	def filter_tokens(tokens):
		filtered = [token for token in tokens if token not in stopwords]
		filtered = [re_num_simple.sub("<num>", token) for token in filtered]
		return filtered

	def filter_tokens_body(body):
		filtered = [token for token in tokens if token not in stopwords] for tokens in body
		filtered = [[re_num_simple.sub("<num>", token) for token in tokens] for tokens in filtered]
		filtered = [tokens for tokens in filtered if tokens != []]
		return filtered

	# delete tokens that are either stopwords or puncuation (dont add significant info)
	# replace numbers with <num> token
	print("Filtering title tokens ...")
	df["title_tokens"] = df.title_tokens.progress_apply(filter_tokens)

	print("Filtering body tokens ...")
	df["body_tokens"] = df.body_tokens.progress_apply(filter_tokens)

	# create columns that contain the number of tokens
	df["n_tokens_title"] = df.title_tokens.apply(len)
	df["n_tokens_body"] = df.body_tokens.apply(len)

	# sort from smallest to largest articles based on the number of body tokens
	df.sort_values("n_tokens_body", axis = 0, inplace = True)

	return df

def split_dataset(df, valid_size, test_size):

	l = len(df)
	idx = np.random.permutation(range(l))

	df_valid = df.iloc[idx[:valid_size], :]
	df_test = df.iloc[idx[valid_size: valid_size + test_size], :]
	df_train = df.iloc[idx[valid_size + test_size:], :]

	return df_train, df_valid, df_test

##########################################################################
##########################################################################
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = "./../data/") #"C:/Users/ioann/Datasets/Hyperpartisan/"
parser.add_argument('--file_name', type = str, default = "training-byarticle-20181122.txt")
parser.add_argument('--usedata', type = int, default = None)
parser.add_argument('--lowercase', type = bool, default = True)
parser.add_argument('--filter_title', type = bool, default = True)
parser.add_argument('--filter_body', type = bool, default = True)
parser.add_argument('--valid_size', type = int, default = 64)
parser.add_argument('--test_size', type = int, default = 64)
args, unparsed = parser.parse_known_args()

print("Processing {} set".format(data_set))

df = load_data(args.data_path, args.file_name, args.filter_body, args.filter_title, args.usedata)
print("{} set loaded with size {}".format(data_set, len(df)))

true_percent = df.hyperpartisan.mean() * 100
print("hyperpartisan statistics: True = {:.2f}% || False = {:.2f}%".format(true_percent, 100 - true_percent))

df = clean_data(df)

df = process_data(df, args.lowercase)

df = filter_data(df)

df_train, df_valid, df_test = split_dataset(df, args.valid_size, args.test_size)

df_train.to_csv(args.data_path + "train_byart.txt", sep = "\t")
df_valid.to_csv(args.data_path + "valid_byart.txt", sep = "\t")
df_test.to_csv(args.data_path + "test_byart.txt", sep = "\t")








