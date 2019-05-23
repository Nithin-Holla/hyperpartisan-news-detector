import argparse
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors

from enums.elmo_model import ELMoModel
from model.JointModel import JointModel
from constants import Constants

from enums.training_mode import TrainingMode
from datasets.hyperpartisan_dataset import HyperpartisanDataset
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan


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

    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             hidden_dim=argument_parser.hidden_dim,
                             num_layers=argument_parser.num_layers,
                             sent_encoder_dropout_rate=.0,
                             doc_encoder_dropout_rate=.0,
                             output_dropout_rate=.0,
                             device=device,
                             skip_connection=argument_parser.skip_connection).to(device)

    checkpoint = torch.load(argument_parser.model_checkpoint, map_location="cpu")
    joint_model.load_state_dict(checkpoint['model_state_dict'])

    return joint_model


def create_hyperpartisan_loaders(argument_parser, glove_vectors):
    hyperpartisan_test_dataset = HyperpartisanDataset(argument_parser.txt_file,
                                                      argument_parser.concat_glove,
                                                      glove_vectors,
                                                      argument_parser.elmo_model)

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
                                                     batch_num_sent, batch_sent_lengths, batch_feat),
                                      task=TrainingMode.Hyperpartisan)

    return predictions.round().long().tolist()

	# return predictions.round().long().tolist()
	return predictions.tolist()

def iterate_metaphor(joint_model, metaphor_data, device):
    batch_inputs = metaphor_data[0].to(device).float()
    batch_lengths = metaphor_data[1].to(device)

    predictions = joint_model.forward(batch_inputs, batch_lengths, task=TrainingMode.Metaphor)

    predictions = predictions.tolist()

    metaphorical = []
    for sent in predictions:
        sent = [p[0] for p in sent]
        metaphorical.append(sum(sent) / len(sent))
    metaphorical = sum(metaphorical) / len(metaphorical)

	metaphorical = []
	for i, sent in enumerate(predictions):
		sent = [int(p[0] > 0.5) for p in sent[:batch_lengths[i].item()]]
		metaphorical.append(int(sum(sent) > 1))
	metaphorical = sum(metaphorical)/len(metaphorical)


def forward_full_hyperpartisan(
        joint_model: JointModel,
        dataloader: DataLoader,
        device: torch.device):
    all_predictions = []
    all_ids = []

    for step, (
    batch_inputs, _, batch_recover_idx, batch_num_sent, batch_sent_lengths, batch_feat, batch_ids) in enumerate(
            dataloader):
        batch_predictions = iterate_hyperpartisan(
            joint_model=joint_model,
            batch_inputs=batch_inputs,
            batch_recover_idx=batch_recover_idx,
            batch_num_sent=batch_num_sent,
            batch_sent_lengths=batch_sent_lengths,
            batch_feat=batch_feat,
            device=device)

        print(batch_ids[0], batch_predictions[0])

        all_predictions.append(batch_predictions[0])
        all_ids.append(batch_ids[0])

    return all_predictions, all_ids


def forward_through_metaphor(joint_model, dataloader, device):
    all_metaphors = []
    all_ids = []

    for step, (
    batch_inputs, _, batch_recover_idx, batch_num_sent, batch_sent_lengths, batch_feat, batch_ids) in enumerate(
            dataloader):
        metaphorical = iterate_metaphor(joint_model, (batch_inputs, batch_sent_lengths), device)

        all_metaphors.append(metaphorical)
        all_ids.append(batch_ids[0])

        print(batch_ids[0], metaphorical)

    return all_metaphors, all_ids


def eval_model(argument_parser):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load GloVe vectors
    if argument_parser.concat_glove:
        glove_vectors = load_glove_vectors(argument_parser)
    else:
        glove_vectors = None

    # Load pre-trained model
    joint_model = initialize_model(argument_parser, device)

    # initialize test set loader
    hyperpartisan_test_dataloader = create_hyperpartisan_loaders(argument_parser, glove_vectors)

    joint_model.eval()

    if argument_parser.mode == "normal":

        with torch.no_grad():
            predictions, ids = forward_full_hyperpartisan(joint_model, hyperpartisan_test_dataloader, device)

        with open("output.txt", "w") as f:
            for Id, prediction in zip(ids, predictions):
                f.write(Id + " " + str(prediction == 1) + "\n")

		# with open("output_hyper.txt", "w") as f:
		# 	for Id, prediction in zip(ids, predictions):
		# 		f.write(Id + " " + str(prediction == 1) + "\n")
		with open("prob_joint.txt", "w") as f:
			for Id, prob in zip(ids, predictions):
				f.write(Id + " " + str(prob) + "\n")

        with torch.no_grad():
            metaphorical, ids = forward_through_metaphor(joint_model, hyperpartisan_test_dataloader, device)

        with open("metaphorical.txt", "w") as f:
            for Id, meta in zip(ids, metaphorical):
                f.write(Id + " " + str(meta) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to save/load the model')
    parser.add_argument('--vector_file_name', type=str, default='glove.840B.300d.txt',
                        help='File in which vectors are saved')
    parser.add_argument('--vector_cache_dir', type=str, default='vector_cache',
                        help='Directory where vectors would be cached')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training the model')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of the recurrent network')
    parser.add_argument('--glove_size', type=int, default=10000,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--txt_file', type=str, default="",
                        help='text file name containing the processed xml file')
    parser.add_argument('--hdf5_file', type=str, default="",
                        help='hdf5 file name that contains the elmo embeddings')
    parser.add_argument('--elmo_vector', type=str, choices=["top", "average"], default="average",
                        help='method for final emlo embeddings used')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers to be used in the biLSTM sentence encoder')
    parser.add_argument('--mode', type=str, choices=["normal", "metaphorical"], default="metaphorical")
    parser.add_argument('--skip_connection', action='store_true',
                        help='Indicates whether a skip connection is to be used in the sentence encoder '
                             'while training on hyperpartisan task')
    parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
                        help='ELMo model from which vectors are used')
    parser.add_argument('--concat_glove', action='store_true',
                        help='Whether GloVe vectors have to be concatenated with ELMo vectors for words')

    argument_parser = parser.parse_args()

    eval_model(argument_parser)
