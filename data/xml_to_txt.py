import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from os import listdir
import argparse

def xml_parser(data_path, xml_file_articles, xml_file_ground_truth, chunk_size):

	# initialize dataframe for outout
	columns = ["published-at", "title", "body", "hyperpartisan", "bias", "url", "labeled-by"]
	df = pd.DataFrame(columns = columns)
	
	body_tags = ["a", "p", "q"]
	
	txt_file = xml_file_articles.split("-", maxsplit = 1)[1].split(".")[0] + ".txt"

	n_items = 0
	
	bias_dict = {"left": 0, "left-center": 1, "least": 2, "right-center": 3, "right": 4}
	
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
				
				df_elem = pd.DataFrame([[Date, Title, "", 0, "", "", ""]], index = [Id], columns = columns)
				start_of_body = ""
				
			# finalize this article and append it to the parent dataframe
			else:
								
				df_elem.loc[Id, "body"] = df_elem.loc[Id, "body"].replace("[...]", "").replace("  ", " ")	

				# get ground truth info for this article from the other iterator (article ids are in order, checked)
				_, ground_truth_elem = next(ground_truth_iter)
				
				df_elem.loc[Id, "hyperpartisan"] = ground_truth_elem.attrib["hyperpartisan"] == "true"
				df_elem.loc[Id, "bias"] = bias_dict[ground_truth_elem.attrib["bias"]]
				df_elem.loc[Id, "url"] = ground_truth_elem.attrib["url"]
				df_elem.loc[Id, "labeled-by"] = ground_truth_elem.attrib["labeled-by"]
				
				df = df.append(df_elem)
				
				n_items += 1
			
		# append this text to the article body	
		elif article_elem.tag in body_tags:
			
			if article_event == "start":
				if article_elem.text is not None:
					if "&#" not in article_elem.text:
						df_elem.loc[Id, "body"] += start_of_body + article_elem.text
						start_of_body = " "
						
		
		# write processed data		
		if n_items == chunk_size:
			
			if txt_file not in listdir(data_path):
				df.to_csv(data_path + txt_file, sep = "\t")
				
			else:
				df.to_csv(data_path + txt_file, sep = "\t", mode = "a", header = False)
			
			df = pd.DataFrame(columns = columns)
			n_items = 0
					 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = "")
parser.add_argument('--xml_file_articles', type = str, default = "")
parser.add_argument('--xml_file_ground_truth', type = str, default = "./../data/")
parser.add_argument('--chunk_size', type = int, default = 50000)
args, unparsed = parser.parse_known_args()

xml_parser(args.data_path, args.xml_file_articles, args.xml_file_ground_truth, args.chunk_size)

