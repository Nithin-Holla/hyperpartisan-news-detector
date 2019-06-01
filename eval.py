import argparse
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors

from enums.elmo_model import ELMoModel
from model.JointModel import JointModel
from constants import Constants

from enums.training_mode import TrainingMode
from datasets.hyperpartisan_dataset import HyperpartisanDataset
from datasets.metaphor_dataset import MetaphorDataset
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.data_helper import DataHelper
from helpers.metaphor_loader import MetaphorLoader
from helpers.hyperpartisan_loader import HyperpartisanLoader
from batches.hyperpartisan_batch import HyperpartisanBatch
from model.Ensemble import Ensemble

import os
from sklearn import metrics

def load_glove_vectors(argument_parser):
	glove_vectors = Vectors(name=argument_parser.vector_file_name,
							cache=argument_parser.vector_cache_dir,
							max_vectors=argument_parser.glove_size)
	glove_vectors.stoi = {k: v + 2 for (k, v) in glove_vectors.stoi.items()}
	glove_vectors.itos = ['<unk>', '<pad>'] + glove_vectors.itos
	glove_vectors.stoi['<unk>'] = 0
	glove_vectors.stoi['<pad>'] = 1
	unk_vector = torch.zeros((1, glove_vectors.dim))
	pad_vector = torch.mean(glove_vectors.vectors, dim=0, keepdim=True)
	glove_vectors.vectors = torch.cat((unk_vector, pad_vector, glove_vectors.vectors), dim=0)

	return glove_vectors


def initialize_model(argument_parser, device):

	if argument_parser.elmo_model == ELMoModel.Original:
		elmo_vectors_size = Constants.ORIGINAL_ELMO_EMBEDDING_DIMENSION
	elif argument_parser.elmo_model == ELMoModel.Small:
		elmo_vectors_size = Constants.SMALL_ELMO_EMBEDDING_DIMENSION

	if argument_parser.concat_glove:
		total_embedding_dim = elmo_vectors_size + Constants.GLOVE_EMBEDDING_DIMENSION

	if argument_parser.model_type == "ensemble":

		assert not os.path.isfile(argument_parser.checkpoint_path)
		model = Ensemble(path_to_models=argument_parser.checkpoint_path,
						 sent_encoder_hidden_dim=argument_parser.sent_encoder_hidden_dim,
						 doc_encoder_hidden_dim=argument_parser.doc_encoder_hidden_dim,
						 num_layers=argument_parser.num_layers,
						 skip_connection=argument_parser.skip_connection,
						 include_article_features=argument_parser.include_article_features,
						 document_encoder_model=argument_parser.document_encoder_model,
						 pre_attention_layer=argument_parser.pre_attention_layer,
						 total_embedding_dim=total_embedding_dim,
						 device=device
						 )

	else:

		assert os.path.isfile(argument_parser.checkpoint_path)
		model = JointModel(embedding_dim=total_embedding_dim,
							 sent_encoder_hidden_dim=argument_parser.sent_encoder_hidden_dim,
							 doc_encoder_hidden_dim=argument_parser.doc_encoder_hidden_dim,
							 num_layers=argument_parser.num_layers,
							 sent_encoder_dropout_rate=0.,
							 doc_encoder_dropout_rate=0.,
							 output_dropout_rate=0.,
							 device=device,
							 skip_connection=argument_parser.skip_connection,
							 include_article_features=argument_parser.include_article_features,
							 doc_encoder_model=argument_parser.document_encoder_model,
							 pre_attn_layer=argument_parser.pre_attention_layer
							 ).to(device)

		checkpoint = torch.load(argument_parser.checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])

	return model


def create_hyperpartisan_loaders(argument_parser, glove_vectors):

	hyperpartisan_train_dataset, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
		hyperpartisan_dataset_folder=argument_parser.hyperpartisan_dataset_folder,
		concat_glove=argument_parser.concat_glove,
		glove_vectors=glove_vectors,
		elmo_model=argument_parser.elmo_model)

	hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
		train_dataset=hyperpartisan_train_dataset,
		validation_dataset=hyperpartisan_validation_dataset,
		test_dataset=None,
		batch_size=argument_parser.batch_size,
		shuffle=False)

	return hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader


def create_metaphor_loaders(argument_parser, glove_vectors):

	metaphor_train_dataset, metaphor_validation_dataset, metaphor_test_dataset = MetaphorLoader.get_metaphor_datasets(
		metaphor_dataset_folder=argument_parser.metaphor_dataset_folder,
		concat_glove=argument_parser.concat_glove,
		glove_vectors=glove_vectors,
		elmo_model=argument_parser.elmo_model,
		lowercase_sentences=False,
		tokenize_sentences=True,
		only_news=False)

	metaphor_train_dataloader, metaphor_validation_dataloader, metaphor_test_dataloader = DataHelper.create_dataloaders(
		train_dataset=metaphor_train_dataset,
		validation_dataset=metaphor_validation_dataset,
		test_dataset=metaphor_test_dataset,
		batch_size=argument_parser.batch_size,
		shuffle=False)

	return metaphor_train_dataloader, metaphor_validation_dataloader, metaphor_test_dataloader


def iterate_hyperpartisan(
		model,
		batch_inputs,
		batch_recover_idx,
		batch_num_sent,
		batch_sent_lengths,
		batch_feat,
		device,
		output):
	batch_inputs = batch_inputs.to(device)
	batch_recover_idx = batch_recover_idx.to(device)
	batch_num_sent = batch_num_sent.to(device)
	batch_sent_lengths = batch_sent_lengths.to(device)
	batch_feat = batch_feat.to(device)

	predictions = model.forward(batch_inputs, (batch_recover_idx, batch_num_sent, batch_sent_lengths, batch_feat),
							  task=TrainingMode.Hyperpartisan)


	if output == "predictions":
		return predictions.round().long().tolist()
	else:
		return predictions.tolist()

def iterate_hyperpartisan_through_metaphor(model, metaphor_data, device):
	batch_inputs = metaphor_data[0].to(device).float()
	batch_lengths = metaphor_data[1].to(device)

	predictions = model.forward(batch_inputs, batch_lengths, task=TrainingMode.Metaphor)
	predictions = predictions.tolist()

	metaphorical = []
	for i, sent in enumerate(predictions):
		sent = [int(p > 0.5) for p in sent[:batch_lengths[i].item()]]
		metaphorical.append(int(sum(sent) > 0))
	metaphorical = sum(metaphorical)/len(metaphorical)

	return metaphorical


def iterate_metaphor(model, metaphor_data, device, output):
	batch_inputs = metaphor_data[0].to(device).float()
	batch_targets = metaphor_data[1].to(device).view(-1).float()
	batch_lengths = metaphor_data[2].to(device)

	predictions = model.forward(batch_inputs, batch_lengths, task=TrainingMode.Metaphor)

	unpadded_targets = batch_targets[batch_targets != -1]
	unpadded_predictions = predictions.view(-1)[batch_targets != -1]

	if output == "predictions":
		return unpadded_targets.long().tolist(), unpadded_predictions.round().long().tolist()
	else:
		return unpadded_targets.long().tolist(), unpadded_predictions.tolist()


def forward_full_metaphor(model, dataloader, device, output):

	all_targets = []
	all_predictions = []

	for step, metaphor_data in enumerate(dataloader):

		batch_targets, batch_predictions = iterate_metaphor(model, metaphor_data, device, output)

		all_targets.extend(batch_targets)
		all_predictions.extend(batch_predictions)

	return all_targets, all_predictions


def forward_full_hyperpartisan(model, dataloader, device, output):

	all_predictions = []
	all_ids = []

	for step, (batch_inputs, _, batch_num_sent, batch_sent_lengths, batch_feat, batch_ids) in enumerate(dataloader):

		hyperpartisan_batch = HyperpartisanBatch(10000)
		hyperpartisan_batch.add_data(batch_inputs, 0, batch_num_sent, batch_sent_lengths, batch_feat)
		hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()

		batch_predictions = iterate_hyperpartisan(
			model=model,
			batch_inputs=hyperpartisan_data[0],
			batch_recover_idx=hyperpartisan_data[2],
			batch_num_sent=hyperpartisan_data[3],
			batch_sent_lengths=hyperpartisan_data[4],
			batch_feat=hyperpartisan_data[5],
			device=device,
			output=output)

		all_predictions.append(batch_predictions)
		all_ids.append(batch_ids[0])

	return all_predictions, all_ids



def forward_hyperpartisan_through_metaphor(model, dataloader, device):
	all_metaphors = []
	all_ids = []

	for step, (batch_inputs, _, batch_num_sent, batch_sent_lengths, batch_feat, batch_ids) in enumerate(dataloader):

		metaphor_batch = HyperpartisanBatch(10000)
		metaphor_batch.add_data(batch_inputs, 0, batch_num_sent, batch_sent_lengths, batch_feat)
		metaphor_data = metaphor_batch.pad_and_sort_batch()

		metaphorical = iterate_hyperpartisan_through_metaphor(model, (metaphor_data[0], metaphor_data[4]), device)

		all_metaphors.append(metaphorical)
		all_ids.append(batch_ids[0])

	return all_metaphors, all_ids


def eval_model(argument_parser):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	try:
		os.mkdir("model_output")
	except:
		pass

	# Load GloVe vectors
	if argument_parser.concat_glove:
		glove_vectors = load_glove_vectors(argument_parser)
	else:
		glove_vectors = None

	# Load pre-trained model
	model = initialize_model(argument_parser, device)
	model.eval()

	if argument_parser.mode == "hyperpartisan":

		if argument_parser.testing_on == "training":
			hyperpartisan_dataloader = create_hyperpartisan_loaders(argument_parser, glove_vectors)[0]
		elif argument_parser.testing_on == "validation":
			hyperpartisan_dataloader = create_hyperpartisan_loaders(argument_parser, glove_vectors)[1]

		with torch.no_grad():
			predictions, ids = forward_full_hyperpartisan(model, hyperpartisan_dataloader, device, argument_parser.output)

		if argument_parser.output == "predictions":
			with open("model_output/pred_{}_{}.txt".format(argument_parser.mode, argument_parser.testing_on), "w") as f:
				for Id, prediction in zip(ids, predictions):
					f.write(Id + " " + str(prediction == 1) + "\n")
		else:
			with open("model_output/prob_{}_{}.txt".format(argument_parser.mode, argument_parser.testing_on), "w") as f:
				for Id, prob in zip(ids, predictions):
					f.write(Id + " " + str(prob) + "\n")

	elif argument_parser.mode == "hyper_through_metaphor":

		if argument_parser.testing_on == "training":
			hyperpartisan_dataloader = create_hyperpartisan_loaders(argument_parser, glove_vectors)[0]
		elif argument_parser.testing_on == "validation":
			hyperpartisan_dataloader = create_hyperpartisan_loaders(argument_parser, glove_vectors)[1]

		with torch.no_grad():
			metaphorical, ids = forward_hyperpartisan_through_metaphor(model, hyperpartisan_dataloader, device)

		with open("model_output/metaphorical_hyperpartisan_{}.txt".format(argument_parser.testing_on), "w") as f:
			for Id, meta in zip(ids, metaphorical):
				f.write(Id + " " + str(meta) + "\n")

	elif argument_parser.mode == "metaphor":

		if argument_parser.testing_on == "training":
			metaphor_dataloader = create_metaphor_loaders(argument_parser, glove_vectors)[0]
		elif argument_parser.testing_on == "validation":
			metaphor_dataloader = create_metaphor_loaders(argument_parser, glove_vectors)[1]
		elif argument_parser.testing_on == "test":
			metaphor_dataloader = create_metaphor_loaders(argument_parser, glove_vectors)[2]

		with torch.no_grad():
			targets, predictions = forward_full_metaphor(model, metaphor_dataloader, device, argument_parser.output)

		with open("model_output/metaphor_{}_{}.txt".format(argument_parser.output, argument_parser.testing_on), "w") as f:
			for target, prediction in zip(targets, predictions):
				f.write(str(target) + " " + str(prediction) + "\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_path', type=str, required=True)
	parser.add_argument('--vector_file_name', type=str, default='glove.840B.300d.txt',
						help='File in which vectors are saved')
	parser.add_argument('--vector_cache_dir', type=str, required=True,
						help='Directory where vectors would be cached')
	parser.add_argument('--batch_size', type=int, default=1,
						help='Batch size for training the model')
	parser.add_argument('--sent_encoder_hidden_dim', type=int, default=Constants.DEFAULT_HIDDEN_DIMENSION,
						help='Hidden dimension of the recurrent network')
	parser.add_argument('--doc_encoder_hidden_dim', type=int, default=Constants.DEFAULT_DOC_ENCODER_DIM,
						help='Hidden dimension of the recurrent network')
	parser.add_argument('--glove_size', type=int,
						help='Number of GloVe vectors to load initially')
	parser.add_argument('--hyperpartisan_dataset_folder', type=str,
						help='Path to the hyperpartisan dataset')
	parser.add_argument('--metaphor_dataset_folder', type=str)
	parser.add_argument('--elmo_vector', type=str, choices=["top", "average"], default="average",
						help='method for final emlo embeddings used')
	parser.add_argument('--num_layers', type=int, default=Constants.DEFAULT_NUM_LAYERS,
						help='Number of layers to be used in the biLSTM sentence encoder')
	parser.add_argument('--skip_connection', action='store_true',
						help='Indicates whether a skip connection is to be used in the sentence encoder '
							 'while training on hyperpartisan task')
	parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
						help='ELMo model from which vectors are used')
	parser.add_argument('--concat_glove', action='store_true',
						help='Whether GloVe vectors have to be concatenated with ELMo vectors for words')
	parser.add_argument('--include_article_features', action='store_true',
						help='Whether to append handcrafted article features to the hyperpartisan fc layer')
	parser.add_argument('--document_encoder_model', type=str, default="GRU")
	parser.add_argument('--pre_attention_layer', action="store_true")
	parser.add_argument('--model_type', type=str, choices=["ensemble", "single"], default = "ensemble")
	parser.add_argument('--output', type=str, choices = ["predictions", "probabilities"], default = "predictions",
						help="whether to return the predictions or the probabilities of the instances")
	parser.add_argument('--mode', type=str, choices=["hyperpartisan", "hyper_through_metaphor", "metaphor"], default="hyperpartisan")
	parser.add_argument('--testing_on', type=str, choices=["training", "validation", "test"], default="test")

	argument_parser = parser.parse_args()

	if argument_parser.mode in ["hyper_through_metaphor", "metaphor"]:
		assert "joint" in argument_parser.checkpoint_path

	if argument_parser.mode in ["hyperpartisan", "hyper_through_metaphor"]:
		assert "test" != argument_parser.testing_on

	eval_model(argument_parser)
