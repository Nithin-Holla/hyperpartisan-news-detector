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

	# last row in both files is wrong
	if "validation" in file_name:
		skiprows = 150000 + 1
	else:
		skiprows = 600000 + 1

	# index is the unique article id
	# cols 1, 6 and 7 are published-at, url and labeled-by
	dtype = {"title": str, "body": str, "hyperpartisan": int, "bias": int}
	df = pd.read_csv(data_path + file_name, sep = "\t", index_col = 0, encoding = "ISO-8859-1", dtype = dtype, nrows = usedata, 
		skiprows = [skiprows], usecols = [0, 2, 3, 4, 5])
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
		text = text.replace(" _", " ")
		text = text.replace("–", "-")
		text = text.replace(". . .", "...")
		return text

	# def fix_bad_parsing(text):
	# 	text = text.replace("?", "")
	# 	# text = text.replace("''", '"')
	# 	return text

	# clean text from xml special characters and other inconsistancies
	df["title"] = df.title.apply(clean_text)
	df["body"] = df.body.apply(clean_text)

	# find instancies where apostrophe is parsed as question mark
	# convient way: look for 2 consequtive question marks in the body
	# idx_fix = df.body.apply(lambda x: "??" in x)
	# print("Found {} instancies with bad parsing".format(idx_fix.sum()))

	# # replace question marks with apostrophes in these instancies
	# df.loc[idx_fix, "title"] = df.loc[idx_fix, "title"].apply(fix_bad_parsing)
	# df.loc[idx_fix, "body"] = df.loc[idx_fix, "body"].apply(fix_bad_parsing)

	return df


def process_data(df, lowercase):

	def sent_to_word_tokens(text):
		df_text = pd.Series(text)
		return df_text.apply(lambda x: ["<s>"] + word_tokenize(x) + ["</s>"]).sum()

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
	# adds <s> and </s> tokens to indicate beginning and ending of each sentence
	print("Body to word tokens ...")
	df["body_tokens"] = df.body_sent.progress_apply(sent_to_word_tokens)

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

	punctuation = '.,:;?!"\'\'``+={}[]()#~$--'

	for p in list(punctuation):
		stopwords.add(p)

	# print(stopwords)

	re_num_simple = re.compile('^-?[0-9.,]+([eE^][0-9]+)?(th)?$')

	def filter_tokens(tokens):
		filtered = [token for token in tokens if token not in stopwords]
		filtered = [re_num_simple.sub("<num>", token) for token in filtered]
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


def remove_small_big_articles(df, minimum_len, maximum_len):
	len_before = len(df)
	if minimum_len:
		df = df.loc[df.n_sent_body >= minimum_len, :]
	if maximum_len:
		df = df.loc[df.n_sent_body < maximum_len, :]
	chng = (len_before - len(df)) / len_before * 100
	print("Removed small articles. Size reduced from {} to {} ({:.2f}%)".format(len_before, len(df), chng))
	return df


def remove_duplicate_tags(df):
# by removing some tokens
# there are instances where there appears to be empty sentences
# (consequtive <s> </s> tags)

	def rmv_dpl(x):
		Idx = []
		for idx, e1, e2 in zip(range(len(x)-1), x[:-1], x[1:]):
			if e1 == "<s>" and e2 == "</s>":
				Idx.append(idx)
		for idx in Idx:
			x = x[:idx] + x[idx + 2:]
		return x

	print("Removing duplicates tags ...")
	df["body_tokens"] = df.body_tokens.progress_apply(rmv_dpl)

	return df


def get_sentence_indices(df):
# creates a new column that contains a list indicating the indices where each sentence begins

	def get_sent_idx(tokens):
		return [i for i, token in enumerate(tokens) if token == "<s>"]

	print("Getting sentence indices ...")
	df["sent_idx"] = df.body_tokens.progress_apply(get_sent_idx)

	return df


def split_dataset(df, ratio):

	l = len(df)

	idx = np.random.permutation(range(l))

	splt = int(ratio * l)

	df_valid = df.iloc[idx[:splt], :]
	df_test = df.iloc[idx[splt:], :]

	return df_valid, df_test

##########################################################################
##########################################################################
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = "./../data/") #"C:/Users/ioann/Datasets/Hyperpartisan/"
parser.add_argument('--file_name', type = str, default = "training-bypublisher-20181122.txt")
parser.add_argument('--usedata', type = int, default = None)
parser.add_argument('--lowercase', type = bool, default = False)
parser.add_argument('--filter_title', type = bool, default = True)
parser.add_argument('--filter_body', type = bool, default = True)
parser.add_argument('--minimum_len', type = int, default = 4)
parser.add_argument('--maximum_len', type = int, default = 70)
parser.add_argument('--split_ratio', type = float, default = 0.5)
args, unparsed = parser.parse_known_args()

if "validation" in args.file_name:
	data_set = "validation"
else:
	data_set = "training"

print("Processing {} set".format(data_set))

df = load_data(args.data_path, args.file_name, args.filter_body, args.filter_title, args.usedata)
print("{} set loaded with size {}".format(data_set, len(df)))

true_percent = df.hyperpartisan.mean() * 100
print("hyperpartisan statistics: True = {:.2f}% || False = {:.2f}%".format(true_percent, 100 - true_percent))

left = (df.bias == 0).mean() * 100
left_center = (df.bias == 1).mean() * 100
least = (df.bias == 2).mean() * 100
right_center = (df.bias == 3).mean() * 100
right = (df.bias == 4).mean() * 100
print("bias statistics: left = {:.2f}% || left-center = {:.2f}% || least = {:.2f}% || right-center = {:.2f}% || right = {:.2f}%".format(
	left, left_center, least, right_center, right))

df = clean_data(df)

df = process_data(df, args.lowercase)

df = remove_small_big_articles(df, args.minimum_len, args.maximum_len)

df = filter_data(df)

df = remove_duplicate_tags(df)

df = get_sentence_indices(df)

if data_set == "validation":
	df_valid, df_test = split_dataset(df, args.split_ratio)
	print("Validation set split with ratio {} to valid and test sets".format(args.split_ratio))

	df_valid.to_csv(args.data_path + "valid" + ".txt", sep = "\t")
	df_test.to_csv(args.data_path + "test" + ".txt", sep = "\t")
	print("valid and test set saved succesfully.")	

else:

	df.to_csv(args.data_path + "train" + ".txt", sep = "\t")
	print("train set saved succesfully.")	








