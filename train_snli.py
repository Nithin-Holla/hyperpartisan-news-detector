import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module, CrossEntropyLoss
from torchtext.vocab import Vectors

import argparse
import os

from constants import Constants
from datasets.snli_dataset import SnliDataset
from helpers.snli_loader import SnliLoader
from helpers.data_helper_snli import DataHelperSnli
from helpers.utils_helper import UtilsHelper

from model.SnliClassifer import MLP

from datetime import datetime
import time

from tensorboardX import SummaryWriter

utils_helper = UtilsHelper()

def initialize_model(argument_parser, device, glove_vectors_dim):

	total_embedding_dim = Constants.DEFAULT_ELMO_EMBEDDING_DIMENSION + glove_vectors_dim

	model = MLP(argument_parser, total_embedding_dim, device).to(device)

	### WEIGHT DECAY SEPERATELLY????
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
					   lr=argument_parser.learning_rate, weight_decay=argument_parser.weight_decay)

	# Load the checkpoint if found
	start_epoch = 1

	if argument_parser.load_model and os.path.isfile(argument_parser.model_checkpoint):
		checkpoint = torch.load(argument_parser.model_checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if checkpoint['epoch']:
			start_epoch = checkpoint['epoch'] + 1

		print('Found previous model state')
	else:
		print('Loading model state...Done')

	print("Starting training in '%s' mode from epoch %d..." %
		  (argument_parser.mode, start_epoch))

	return model, optimizer, epoch

def create_loaders(argument_parser, glove_vectors):

	train_dataset, validation_dataset, test_dataset = SnliLoader.get_snli_datasets(snli_dataset_folder, glove_vectors)

	train_dataloader, validation_dataloader, test_dataloader = DataHelperSnli.create_dataloaders(
		train_dataset, validation_dataset, test_dataset, argument_parser.batch_size, shuffle=True)

	return train_dataloader, validation_dataloader. test_dataloader

def iterate_dataset(model, optimizer, criterion, snli_data,	device,	train = False):

	batch_inputs1 = snli_data[0].to(device)
	batch_sent_lengths1 = snli_data[1].to(device)
	batch_recover_idx1 = snli_data[2].to(device)

	batch_inputs2 = snli_data[3].to(device)
	batch_sent_lengths2 = snli_data[4].to(device)
	batch_recover_idx2 = snli_data[5].to(device)

	batch_targets = snli_data[6].to(device)

	if train:

		logits = model.forward(batch_inputs1, batch_inputs2, batch_sent_lengths1, batch_sent_lengths2, batch_recover_idx1, batch_recover_idx2)

		loss = criterion(logits, batch_targets)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	else:

		with torch.no_grad():

			logits = model.forward(batch_inputs1, batch_inputs2, batch_sent_lengths1, batch_sent_lengths2, batch_recover_idx1, batch_recover_idx2)

			loss = criterion(logits, batch_targets)

	accuracy = utils_helper.calculate_accuracy(predictions, batch_targets)

	return loss.item(), accuracy.item(), batch_targets.long().tolist(), torch.argmax(predictions, dim=1).long().tolist()

def forward_full_dataset(model, optimizer, criterion, dataloader, device, train = False):

	all_targets = []
	all_predictions = []

	running_loss = 0
	running_accuracy = 0

	total_length = len(dataloader)
	for step, snli_data in enumerate(dataloader):

		loss, accuracy, batch_targets, batch_predictions = iterate_dataset(model, optimizer, criterion, snli_data, device, train)

		running_loss += loss
		running_accuracy += accuracy
		all_targets += batch_targets
		all_predictions += batch_predictions

	final_loss = running_loss / (step + 1)
	final_accuracy = running_accuracy / (step + 1)

	return final_loss, final_accuracy, all_targets, all_predictions

def train_and_eval_dataset(model, optimizer, criterion, train_dataloader, validation_dataloader, device, best_f1_score,	summary_writer,	epoch):

	model.train()
	loss_train, accuracy_train, _, _ = forward_full_dataset(model, optimizer, criterion, train_dataloader, device, train=True)

	model.eval()
	loss_valid, accuracy_valid, valid_targets, valid_predictions = forward_full_dataset(model, optimizer, criterion, validation_dataloader, device)

	f1, precision, recall = utils_helper.calculate_metrics(valid_targets, valid_predictions)

	log_metrics(summary_writer,	epoch, loss_train, accuracy_train, loss_valid, accuracy_valid, precision, recall, f1)

	print_stats(loss_train, loss_valid, accuracy_train, accuracy_valid, precision, recall, f1, best_f1_score < f1, epoch)

	return f1

def print_stats(train_loss, valid_loss,	train_accuracy,	valid_accuracy,	valid_precision, valid_recall, valid_f1, epoch = None, new_best_score = False):

	epoch_str = str(epoch) if epoch is not None else '_'

	new_best_str = '<- new best result' if new_best_score else ''

	print("[{}] epoch {} || LOSS: train = {:.4f}, valid = {:.4f} || ACCURACY: train = {:.4f}, "
		  "valid = {:.4f} || PRECISION: valid = {:.4f} || RECALL: valid = {:.4f} || F1 SCORE = {:.4f} {}".format(
			  datetime.now().time().replace(microsecond=0), epoch_str, train_loss, valid_loss, train_accuracy, valid_accuracy,
			  valid_precision, valid_recall, valid_f1, new_best_str))

def cache_model(model, optimizer, epoch=None):
	torch_state = {'model_state_dict': model.state_dict(),
				   'optimizer_state_dict': optimizer.state_dict()}

	if epoch is not None:
		torch_state['epoch'] = epoch

	torch.save(torch_state, argument_parser.model_checkpoint)

def log_metrics(summary_writer,	global_step, loss_train, accuracy_train, valid_loss, valid_accuracy, valid_precision, valid_recall,	valid_f1):

	summary_writer.add_scalar('train_loss', loss_train, global_step=global_step)
	summary_writer.add_scalar('train_accuracy', accuracy_train, global_step=global_step)
	summary_writer.add_scalar('valid_loss', valid_loss, global_step=global_step)
	summary_writer.add_scalar('valid_accuracy', valid_accuracy, global_step=global_step)
	summary_writer.add_scalar('valid_precision', valid_precision, global_step=global_step)
	summary_writer.add_scalar('valid_recall', valid_recall, global_step=global_step)
	summary_writer.add_scalar('valid_f1', valid_f1, global_step=global_step)

def train_model(argument_parser):

	# Flags for deterministic runs
	if argument_parser.deterministic:
		utils_helper.initialize_deterministic_mode(argument_parser.deterministic)

	# Set device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Load GloVe vectors
	glove_vectors = utils_helper.load_glove_vectors(argument_parser.vector_file_name, argument_parser.vector_cache_dir, argument_parser.glove_size)

	# Define the model, the optimizer and the loss module
	model, optimizer, start_epoch = initialize_model(argument_parser, device, glove_vectors.dim)

	summary_writer = SummaryWriter(f'runs/exp-snli-lr_{argument_parser.learning_rate}-wd_{argument_parser.weight_decay}')
	
	criterion = CrossEntropyLoss()

	train_dataloader, validation_dataloader, test_dataloader = create_snli_loaders(argument_parser, glove_vectors)

	tic = time.clock()

	best_f1 = .0
	best_epoch = 1

	counter = 0

	for epoch in range(start_epoch, argument_parser.max_epochs + 1):

		f1 = train_and_eval_dataset(model, optimizer, criterion, train_dataloader, validation_dataloader, device, best_f1_score, summary_writer, epoch)

		if f1 > best_f1:
			counter = 0
			best_f1 = f1
			best_epoch = epoch
			cache_model(model, optimizer, epoch)
		else:
			counter += 1

		if counter == 3:
			break

	print("[{}] Training completed in {:.2f} minutes".format(datetime.now().time().replace(microsecond=0), (time.clock() - tic) / 60))
	print("[{}] Loading model of epoch {} with f1 score {:.4f}".format(datetime.now().time().replace(microsecond=0), best_epoch, best_f1))

	summary_writer.close()

	model = MLP(argument_parser, total_embedding_dim, device).to(device)
	model.load_state_dict(checkpoint['model_state_dict'])

	model.eval()
	with torch.no_grad():
		test_loss, test_accuracy, test_targets, test_predictions = forward_full_dataset(model, optimizer, criterion, test_dataloader, device)

	test_f1, test_precision, test_recall = utils_helper.calculate_metrics(test_targets, test_predictions)

	print("[{}] TEST SET: loss = {:.4f}, accu = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(
		datetime.now().time().replace(microsecond=0), test_loss, test_accuracy, test_precision, test_recall, test_f1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True, "checkpoints/snli/encoder_classifer.pt")
    parser.add_argument('--load_model', action='store_true', default = False)
    parser.add_argument('--vector_file_name', type=str, required=True)
    parser.add_argument('--vector_cache_dir', type=str, default=Constants.DEFAULT_VECTOR_CACHE_DIR)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--hidden_dim', type=int, default=Constants.DEFAULT_HIDDEN_DIMENSION)
    parser.add_argument('--glove_size', type=int, default = 10000)
    parser.add_argument('--weight_decay', type=float, default = 0.0)
    parser.add_argument('--snli_dataset_folder', type=str, required = True)
    parser.add_argument('--deterministic', type=int, default = True)
    parser.add_argument('--sent_encoder_dropout_rate', type=float, default=0.)
    parser.add_argument('--num_layers', type=int, default=Constants.DEFAULT_NUM_LAYERS)
    parser.add_argument('--skip_connection', action='store_true', default=Constants.DEFAULT_SKIP_CONNECTION)
    argument_parser = parser.parse_args()

	train_model(argument_parser)