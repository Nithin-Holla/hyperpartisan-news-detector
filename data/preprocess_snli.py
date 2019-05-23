import pandas as pd
from nltk import word_tokenize
import os

def get_snli(snli_folder_path):

	# load original SNLI dataset
	SNLI = {"train": pd.read_csv(snli_folder_path + "/snli_1.0_train.txt", sep = "\t"),
			"dev": pd.read_csv(snli_folder_path + "/snli_1.0_dev.txt", sep = "\t"),
			"test": pd.read_csv(snli_folder_path + "/snli_1.0_test.txt", sep = "\t")}

	for data_set in SNLI.keys():

		# keep only pairs with gold labels and only relevant columns
		SNLI[data_set] = SNLI[data_set][SNLI[data_set].gold_label != "-"][["gold_label", "sentence1", "sentence2"]]

		# remove some examples with non-string values at sentence fields
		valid_indices = SNLI[data_set][["sentence1", "sentence2"]].applymap(type).eq(str).prod(axis = 1).astype(bool)
		SNLI[data_set] = SNLI[data_set].loc[valid_indices, :]
		SNLI[data_set].reset_index(inplace = True, drop = True)

		# map column labels to integers
		# entailment: 0, contradiction: 1, neutral: 2
		SNLI[data_set]["label"] = 0
		SNLI[data_set].loc[SNLI[data_set].gold_label == "neutral", "label"] = 1
		SNLI[data_set].loc[SNLI[data_set].gold_label == "contradiction", "label"] = 2

		del SNLI[data_set]["gold_label"]

		for sset in [1, 2]:

			# tokenize sentences
			SNLI[data_set].loc[:, "sentence" + str(sset)] = SNLI[data_set]["sentence" + str(sset)].apply(word_tokenize)

			# add column for the number of tokens (usefull for batching)
			SNLI[data_set]["ntokens" + str(sset)] = SNLI[data_set]["sentence" + str(sset)].apply(len)

		SNLI[data_set].to_csv(snli_folder_path + "snli_processed/" + data_set + ".txt", sep = "\t")

		print("Finished processing {}".format(data_set))

	return SNLI


def write_sentences(SNLI, snli_folder_path):

	for data_set in SNLI.keys():
		for sset in [1, 2]:

			sentences_filename = snli_folder_path + "snli_processed/" + data_set + "_" + str(sset) + "_elmo.txt"

			with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
				for sentence in SNLI[data_set]["sentence" + str(sset)].apply(lambda x: " ".join(x)).tolist():
					sentences_file.write(f'{sentence}\n')

			print("Finished writing elmo {} {}".format(data_set, sset))


if __name__ == "__main__":

	snli_folder_path = "C:/Users/ioann/Datasets/snli_1.0/"

	if "snli_processed" not in os.listdir(snli_folder_path):
		os.mkdir(snli_folder_path + "snli_processed")

	SNLI = get_snli(snli_folder_path)

	write_sentences(SNLI, snli_folder_path)