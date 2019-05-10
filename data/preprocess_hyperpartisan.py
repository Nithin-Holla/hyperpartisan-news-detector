import pandas as pd
import xml.etree.ElementTree as ET
from nltk import sent_tokenize, word_tokenize
import numpy as np
import re

def url_to_author(url):

	tags = ["www", "m", "video", "news"]
	url = url.strip("https://").strip("http://")
	start = url.split(".")[0]
	if start in tags:
		author = url.split(".")[1]
	else:
		author = url.split(".")[0]
	return author

def clean_text(text):
	# text = re.sub('(?:[-\w.]|(?:%[\da-fA-F]{2}))+?.com', "website", text)
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

def xml_parser(data_path, xml_file_articles, xml_file_ground_truth):

	columns = ["date", "title_tokens", "body_tokens", "hyperpartisan", "author", "length_in_sent", "length_in_words"]
	df = pd.DataFrame(columns = columns)

	body_tags = ["p", "a", "q"]

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

				Body = ""
				
			# finalize this article and append it to the parent dataframe
			else:
								
				# get ground truth info for this article from the other iterator (article ids are in order, checked)
				_, ground_truth_elem = next(ground_truth_iter)
				
				Hyperpartisan = ground_truth_elem.attrib["hyperpartisan"] == "true"
				Author = url_to_author(ground_truth_elem.attrib["url"])

				Body_sent_tokens = sent_tokenize(Body)
				Body_tokens = [word_tokenize(sent) for sent in Body_sent_tokens]

				Length_in_sent = len(Body_sent_tokens)
				Length_in_words = sum(len(sent) for sent in Body_sent_tokens)

				# Body_tokens = clean_websites(Body_tokens)

				df_elem = pd.DataFrame([[Date, Title_tokens, Body_tokens, Hyperpartisan, Author, Length_in_sent, Length_in_words]], index = [Id], columns = columns)
				df = df.append(df_elem)
			
		# append this text to the article body	
		elif article_elem.tag in body_tags:
			
			if article_event == "start":
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

	df_valid = pd.DataFrame()

	for n in range(1, 20, 1):
		df_valid = bysplit(df, df_valid, n)
		if len(df_valid) > 127:
			break

	df_train = df.loc[~df.index.isin(df_valid.index), :]

	df_train = df_train.append(df_valid.loc[df_valid.author == "wthr", :])
	df_valid = df_valid.loc[~(df_valid.author == "wthr"), :]

	print(df_train.hyperpartisan.mean(), len(df_train))
	print(df_valid.hyperpartisan.mean(), len(df_valid))

	return df_train, df_valid

valid_size = 128
data_path = "C:/Users/ioann/Datasets/Hyperpartisan/"
xml_file_articles = "articles-training-byarticle-20181122.xml"
xml_file_ground_truth = "ground-truth-training-byarticle-20181122.xml"

df = xml_parser(data_path, xml_file_articles, xml_file_ground_truth)

df_train, df_valid = split_dataset(df, valid_size)

print(len(set(df_train.author.tolist()).intersection(set(df_valid.author.tolist()))))

df_train = df_train.sort_values("length_in_sent")
df_valid = df_valid.sort_values("length_in_sent")

df_train.to_csv(data_path + "train_byart.txt", sep = "\t")
df_valid.to_csv(data_path + "valid_byart.txt", sep = "\t")
