import argparse
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors
from model.JointModel import JointModel

from datasets.hyperpartisan_dataset import HyperpartisanDataset
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan

def load_glove_vectors(argument_parser):

	glove_vectors = Vectors(name=argument_parser.vector_file_name,
											cache=argument_parser.vector_cache_dir,
											max_vectors=argument_parser.glove_size)
	glove_vectors.stoi = {k: v+2 for (k, v) in glove_vectors.stoi.items()}
	glove_vectors.itos = ['<unk>', '<pad>'] + glove_vectors.itos
	glove_vectors.stoi['<unk>'] = 0
	glove_vectors.stoi['<pad>'] = 1
	unk_vector = torch.zeros((1, glove_vectors.dim))
	pad_vector = torch.mean(glove_vectors.vectors, dim=0, keepdim=True)
	glove_vectors.vectors = torch.cat((unk_vector, pad_vector, glove_vectors.vectors), dim=0)

	return glove_vectors


def initialize_model(argument_parser, device, glove_vectors_dim = 300, elmo_vectors_size = 1024):

	total_embedding_dim = elmo_vectors_size + glove_vectors_dim

	joint_model = JointModel(embedding_dim=total_embedding_dim,
							 hidden_dim=argument_parser.hidden_dim,
							 num_layers=argument_parser.num_layers,
							 sent_encoder_dropout_rate=.0,
							 doc_encoder_dropout_rate=.0,
							 output_dropout_rate=.0,
							 device=device).to(device)

	checkpoint = torch.load(argument_parser.model_checkpoint)
	joint_model.load_state_dict(checkpoint['model_state_dict'])

	return joint_model


def create_hyperpartisan_loaders(argument_parser, glove_vectors):

	hyperpartisan_test_dataset = HyperpartisanDataset(argument_parser.txt_file, glove_vectors)

	_, _, hyperpartisan_test_dataloader = DataHelperHyperpartisan.create_dataloaders(
		train_dataset=None,
		validation_dataset=None,
		test_dataset=hyperpartisan_test_dataset,
		batch_size=argument_parser.batch_size,
		shuffle=False)

	return hyperpartisan_test_dataloader

def iterate_hyperpartisan(
		joint_model,
		batch_inputs,
		batch_targets,
		batch_recover_idx,
		batch_num_sent,
		batch_sent_lengths,
        batch_feat,
		device):

	batch_inputs = batch_inputs.to(device)
	batch_recover_idx = batch_recover_idx.to(device)
	batch_num_sent = batch_num_sent.to(device)
	batch_sent_lengths = batch_sent_lengths.to(device)
	batch_feat = batch_feat.to(device)

	predictions = joint_model.forward(batch_inputs, (batch_recover_idx,
													 batch_num_sent, batch_sent_lengths, batch_feat), task=TrainingMode.Hyperpartisan)

	return predictions.round().long().tolist()

def forward_full_hyperpartisan(
		joint_model: JointModel,
		dataloader: DataLoader,
		device: torch.device):

	all_predictions = []

	for step, (batch_inputs, _, batch_recover_idx, batch_num_sent, batch_sent_lengths, batch_feat) in enumerate(dataloader):

		batch_predictions = iterate_hyperpartisan(
			joint_model=joint_model,
			batch_inputs=batch_inputs,
			batch_recover_idx=batch_recover_idx,
			batch_num_sent=batch_num_sent,
			batch_sent_lengths=batch_sent_lengths,
			batch_feat=batch_feat,
			device=device)

		all_predictions += batch_predictions

	return all_predictions



def eval_model(argument_parser):

	# Set device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Load GloVe vectors
	glove_vectors = load_glove_vectors(argument_parser)

	# Load pre-trained model
	joint_model = initialize_model(argument_parser, device, glove_vectors.dim)

	# initialize test set loader
	hyperpartisan_test_dataloader = create_hyperpartisan_loaders(argument_parser, glove_vectors)

	joint_model.eval()

	with torch.no_grad():

		predictions = forward_full_hyperpartisan(
				joint_model=joint_model,
				dataloader=hyperpartisan_test_dataloader,
				device=device)

	#################
	# do smthing with predictions (not sure yet)
	print(predictions)
	#################


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_checkpoint', type=str, required=True,
						help='Path to save/load the model')
	parser.add_argument('--vector_file_name', type=str, default = 'glove.840B.300d.txt',
						help='File in which vectors are saved')
	parser.add_argument('--vector_cache_dir', type=str, default='vector_cache',
						help='Directory where vectors would be cached')
	parser.add_argument('--batch_size', type=int, default=2,
						help='Batch size for training the model')
	parser.add_argument('--hidden_dim', type=int, default=128,
						help='Hidden dimension of the recurrent network')
	parser.add_argument('--glove_size', type=int, default = 1000,
						help='Number of GloVe vectors to load initially')
	parser.add_argument('--hyperpartisan_dataset_folder', type=str,
						help='Path to the hyperpartisan dataset')
	parser.add_argument('--lowercase', action='store_true', default = False,
						help='Lowercase the sentences before training')
	parser.add_argument('--not_tokenize', action='store_true', default = False,
						help='Do not tokenize the sentences before training')
	parser.add_argument('--txt_file', type = str, default = "train_byart2.txt",
						help='text file name containing the processed xml file')
	parser.add_argument('--hdf5_file', type = str, default = "train_byart_emlo2.hdf5",
						help='hdf5 file name that contains the elmo embeddings')
	parser.add_argument('--elmo_vector', type = str, choices = ["top", "average"], default = "average",
						help='method for final emlo embeddings used')
	parser.add_argument('--num_layers', type = int, default = 1,
						help='Number of layers to be used in the biLSTM sentence encoder')

	argument_parser = parser.parse_args()

	eval_model(argument_parser)

