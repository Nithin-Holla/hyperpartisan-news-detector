import pandas as pd
import xml.etree.ElementTree as ET
from nltk import sent_tokenize, word_tokenize
import numpy as np
from nltk.corpus import stopwords as nltk_stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords

def url_to_author(url):

	tags = ["www", "m", "video", "news"]
	url = url.strip("https://").strip("http://")
	start = url.split(".")[0]
	if start in tags:
		author = url.split(".")[1]
	else:
		author = url.split(".")[0]
	return author

def init_stopwords():

	stopwords = set(nltk_stopwords.words('english'))
	for word in gensim_stopwords:
		stopwords.add(word)
	for word in list(stopwords):
		stopwords.add(word.capitalize())

	punctuation = '.,:;?!"\'\'``+={}[]()#~$--@%^*'
	for p in punctuation:
		stopwords.add(p)
	stopwords.add("''")
	stopwords.add('""')
	stopwords.add("``")
	stopwords.add("...")

	return stopwords

def clean_text(text):
	text = text.replace(".", ". ")
	text = text.replace("=====", "")
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
	text = text.replace("  ", " ")
	text = text.replace(". . . .", "...")
	text = text.replace(". . .", "...")
	return text

def xml_parser(data_path, xml_file_articles, xml_file_ground_truth, remove_stopwords):

	columns = ["date", "title_tokens", "body_tokens", "hyperpartisan", "author", "length_in_par", "length_in_sent", "length_in_words", 
		"links_percent", "quotes_percent", "stopwords_percent", "full_caps_percent", "named_entities_percent",
		"min_sent_length", "mean_sent_length", "max_sent_length"]
	df = pd.DataFrame(columns = columns)

	body_tags = ["p", "a", "q"]

	stopwords = init_stopwords()

	# create iterators
	article_iter = ET.iterparse(data_path + xml_file_articles, events = ("start", "end"))
	ground_truth_iter = ET.iterparse(data_path + xml_file_ground_truth)

	for article_event, article_elem in article_iter:
					   
		if article_elem.tag == "article":
			
			# extract info and init this article (df_elem)
			if article_event == "start":
				
				Id = article_elem.attrib["id"]
				
				try:
					Date = pd.to_datetime(article_elem.attrib["published-at"]).date()
				except:
					Date = np.nan
					
				Title = clean_text(article_elem.attrib["title"])
				Title_tokens = word_tokenize(Title)
				L = len(Title_tokens)

				n_stopwords = sum([1 for token in Title_tokens if token in stopwords])

				n_links = 0
				n_quotes = 0
				n_all = 0
				n_caps = 0
				n_names = 0
				n_paragraphs = 0

				for i, token in enumerate(Title_tokens):
					if token.isupper():
						n_caps += 1
					if i > 0 and token[0].isupper():
						n_names += 1

				Body = ""

			# finalize this article and append it to the parent dataframe
			else:
								
				# get ground truth info for this article from the other iterator (article ids are in order, checked)
				_, ground_truth_elem = next(ground_truth_iter)
				
				Hyperpartisan = ground_truth_elem.attrib["hyperpartisan"] == "true"
				Author = url_to_author(ground_truth_elem.attrib["url"])

				Body_sent_tokens = sent_tokenize(Body)
				L += sum([len(word_tokenize(s)) for s in Body_sent_tokens])

				Body_tokens = [[token for token in word_tokenize(sent)] for sent in Body_sent_tokens]
				n_stopwords += sum([sum([1 for token in sent if token in stopwords]) for sent in Body_sent_tokens])
				sentence_lengths = [len(sent) for sent in Body_sent_tokens]

				min_sent_length = min(sentence_lengths)
				mean_sent_length = sum(sentence_lengths) / len(Body_sent_tokens)
				max_sent_length = max(sentence_lengths)

				for sent in Body_tokens:
					for i, token in enumerate(sent):
						if token.isupper():
							n_caps += 1
						if i > 0 and token[0].isupper():
							n_names += 1

				Title_length = len(Title_tokens)
				Length_in_sent = len(Body_sent_tokens)
				Length_in_words = sum(len(sent) for sent in Body_sent_tokens)

				df_elem = pd.DataFrame([[Date, Title_tokens, Body_tokens, Hyperpartisan, Author, n_paragraphs, Length_in_sent, Length_in_words,
										n_links/n_all, n_quotes/n_all, n_stopwords/(Title_length + Length_in_words), n_caps/(Title_length + Length_in_words),
										n_names/(Title_length + Length_in_words), min_sent_length, mean_sent_length, max_sent_length]],
										index = [Id], columns = columns)
				df = df.append(df_elem)
			
		# append this text to the article body	
		elif article_elem.tag in body_tags:
			
			if article_event == "start":

				if article_elem.tag == "a":
					n_links += 1
				if article_elem.tag == "q":
					n_quotes += 1
				else:
					n_paragraphs += 1
				n_all += 1

				if article_elem.text is not None:
					if "&#" not in article_elem.text:

						Text = clean_text(article_elem.text)
						Body += Text

	df.index = df.index.astype(int)
						
	return df

def split_dataset(df, valid_size):

	def bysplit(df, df_valid, n):
		vc = df.author.value_counts()
		authors = (vc.loc[vc == n]).index.tolist()
		len_authors = len(authors)
		choice = list(set(list(range(0, len_authors, 4))))
		for c in choice:
			df_valid = df_valid.append(df.loc[df.author == authors[c], :])
			if len(df_valid) > 127:
				break

		return df_valid

	df = df.loc[df.index != 410, :]
	df = df.loc[df.index != 95, :]

	# df_valid = pd.DataFrame()

	# for n in range(1, 20, 1):
	# 	df_valid = bysplit(df, df_valid, n)
	# 	if len(df_valid) > 127:
	# 		break

	# df_train = df.loc[~df.index.isin(df_valid.index), :]

	# df_train = df_train.append(df_valid.loc[df_valid.author == "wthr", :])
	# df_valid = df_valid.loc[~(df_valid.author == "wthr"), :]

	path = "C:\\Users\\ioann\\Google Drive\\UvA\\Statistical Methods for Natural Language Semantics\\Research Project\\"
	with open(path + "validation_indices.txt", "r") as f:
		valid_indices = f.read().splitlines()
		for i in range(len(valid_indices)):
			valid_indices[i] = int(valid_indices[i])

	with open(path + "training_indices.txt", "r") as f:
		train_indices = f.read().splitlines()
		for i in range(len(train_indices)):
			train_indices[i] = int(train_indices[i])
			
	df_valid = df.loc[valid_indices, :]
	df_train = df.loc[train_indices, :]

	print(df_train.hyperpartisan.mean(), len(df_train))
	print(df_valid.hyperpartisan.mean(), len(df_valid))

	return df_train, df_valid


def standardize_df(df1, df2):

	df = pd.concat([df1, df2], axis = 0)

	mean_df1 = df1.iloc[:, 5:].mean(axis = 0).values
	std_df1 = df1.iloc[:, 5:].std(axis = 0).values

	mean_df = df.iloc[:, 5:].mean(axis = 0).values
	std_df = df.iloc[:, 5:].std(axis = 0).values

	df1.iloc[:, 5:] = (df1.iloc[:, 5:] - mean_df1) / std_df1
	df2.iloc[:, 5:] = (df2.iloc[:, 5:] - mean_df) / std_df

	return df1, df2

valid_size = 128
data_path = "C:/Users/ioann/Datasets/Hyperpartisan/"
xml_file_articles = "articles-training-byarticle-20181122.xml"
xml_file_ground_truth = "ground-truth-training-byarticle-20181122.xml"
remove_stopwords = True

df = xml_parser(data_path, xml_file_articles, xml_file_ground_truth, remove_stopwords)

df_train, df_valid = split_dataset(df, valid_size)

print(len(set(df_train.author.tolist()).intersection(set(df_valid.author.tolist()))))

# df_train = df_train.sort_values("length_in_sent")
# df_valid = df_valid.sort_values("length_in_sent")

df_train, df_valid = standardize_df(df_train, df_valid)

df_train.to_csv(data_path + "train_byart_new.txt", sep = "\t")
df_valid.to_csv(data_path + "valid_byart_new.txt", sep = "\t")
