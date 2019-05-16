import pandas as pd
import xml.etree.ElementTree as ET
from nltk import sent_tokenize, word_tokenize
import numpy as np
import argparse

def emlo_txtfile(txtfile, title_tokens, body_tokens):

	sentences_filename = txtfile.split(".")[0] + "_elmo.txt"
	print("Writing {} file".format(sentences_filename))

	with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
		for index, article_tokens in enumerate(body_tokens):
			article_title_tokens = title_tokens[index]
			title_text = ' '.join(article_title_tokens)

			sentences_file.write(f'{title_text}\n')

			for sentence_tokens in article_tokens:
				sentence_text = ' '.join(sentence_tokens)

				sentences_file.write(f'{sentence_text}\n')

def xml_parser(argument_parser):

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

	columns = ["date", "title_tokens", "body_tokens", "hyperpartisan", "author", "length_in_sent", "length_in_words", "links_percent", "quotes_percent"]
	df = pd.DataFrame(columns = columns)

	body_tags = ["p", "a", "q"]

	# create iterators
	article_iter = ET.iterparse(argument_parser.xml_file, events = ("start", "end"))

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

				n_links = 0
				n_quotes = 0
				n_all = 0
				
			# finalize this article and append it to the parent dataframe
			else:
				
				Hyperpartisan = 0
				Author = ""

				Body_sent_tokens = sent_tokenize(Body)
				Body_tokens = [word_tokenize(sent) for sent in Body_sent_tokens]

				Length_in_sent = len(Body_sent_tokens)
				Length_in_words = sum(len(sent) for sent in Body_sent_tokens)

				df_elem = pd.DataFrame([[Date, Title_tokens, Body_tokens, Hyperpartisan, Author, Length_in_sent, Length_in_words, n_links/n_all, n_quotes/n_all]], index = [Id], columns = columns)
				df = df.append(df_elem)
			
		# append this text to the article body	
		elif article_elem.tag in body_tags:

			if article_elem.tag == "a":
				n_links += 1
			if article_elem.tag == "q":
				n_quotes += 1
			n_all += 1
			
			if article_event == "start":
				if article_elem.text is not None:
					if "&#" not in article_elem.text:

						Text = clean_text(article_elem.text)
						Body += Text				

	df.index = df.index.astype(int)
						
	return df 

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--xml_file', type=str, default = "pan19-hyperpartisan-news-detection-by-article-test-dataset-2018-12-07.xml",
						help="Xml file name that contains the raw data")
	parser.add_argument('--txt_file', type = str, default = "test_byart.txt",
						help='text file name containing the processed xml file')
	parser.add_argument('--data_size', type = int, default = 2,
						help='Number of articles to load (for debugging)')

	argument_parser = parser.parse_args()

	print("Parsing xml file data...")
	df = xml_parser(argument_parser)
	df.to_csv(argument_parser.txt_file, sep = "\t")
	print("Parsing xml file data... Done")

	print("Creating elmo text file...")
	emlo_txtfile(argument_parser.txt_file, df.title_tokens.tolist(), df.body_tokens.tolist())
	print("Creating elmo text file... Done")