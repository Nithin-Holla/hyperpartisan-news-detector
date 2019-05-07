import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from os import listdir
import argparse

def xml_parser(data_path, xml_file_articles, xml_file_ground_truth):

	# initialize dataframe for outout
	columns = ["published-at", "title", "body", "hyperpartisan", "url", "labeled-by"]
	df = pd.DataFrame(columns = columns)
	
	body_tags = ["a", "p", "q"]
	
	txt_file = xml_file_articles.split("-", maxsplit = 1)[1].split(".")[0] + ".txt"
	
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
					
				Title = article_elem.attrib["title"]
				
				df_elem = pd.DataFrame([[Date, Title, "", 0, "", ""]], index = [Id], columns = columns)
				start_of_body = ""
				
			# finalize this article and append it to the parent dataframe
			else:
								
				df_elem.loc[Id, "body"] = df_elem.loc[Id, "body"].replace("[...]", "").replace("  ", " ")	

				# get ground truth info for this article from the other iterator (article ids are in order, checked)
				_, ground_truth_elem = next(ground_truth_iter)
				
				df_elem.loc[Id, "hyperpartisan"] = ground_truth_elem.attrib["hyperpartisan"] == "true"
				df_elem.loc[Id, "url"] = ground_truth_elem.attrib["url"]
				df_elem.loc[Id, "labeled-by"] = ground_truth_elem.attrib["labeled-by"]
				
				df = df.append(df_elem)
			
		# append this text to the article body	
		elif article_elem.tag in body_tags:
			
			if article_event == "start":
				if article_elem.text is not None:
					if "&#" not in article_elem.text:
						df_elem.loc[Id, "body"] += start_of_body + article_elem.text
						start_of_body = " "
						
	df.to_csv(data_path + txt_file, sep = "\t")
					 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = "./../data/")
parser.add_argument('--xml_file_articles', type = str, default = "articles-training-byarticle-20181122.xml")
parser.add_argument('--xml_file_ground_truth', type = str, default = "ground-truth-training-byarticle-20181122.xml")
args, unparsed = parser.parse_known_args()

xml_parser(args.data_path, args.xml_file_articles, args.xml_file_ground_truth)


