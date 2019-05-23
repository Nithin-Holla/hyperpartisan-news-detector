import argparse
import os

from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import permutation_test
from scipy import stats
from scipy.stats import ttest_ind

import torch
from torch.utils.data import DataLoader

import numpy as np

import sys

from enums.elmo_model import ELMoModel

sys.path.append('..')

from model.JointModel import JointModel
from helpers.utils_helper import UtilsHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.hyperpartisan_loader import HyperpartisanLoader
from datasets.hyperpartisan_dataset import HyperpartisanDataset

from constants import Constants
from enums.training_mode import TrainingMode

utils_helper = UtilsHelper()

def load_model_state(model, model_checkpoint_path):
    if not os.path.isfile(model_checkpoint_path):
        raise Exception('Model checkpoint path is invalid')

    checkpoint = torch.load(model_checkpoint_path)
    if not checkpoint['model_state_dict']:
        raise Exception('Model state dictionary checkpoint not found')

    model.load_state_dict(checkpoint['model_state_dict'])

def initialize_models(
        hyperpartisan_model_checkpoint_path: str,
        joint_model_checkpoint_path: str,
        device: torch.device,
        elmo_model: ELMoModel,
        glove_vectors_dim: int):

    print('Loading model state...\r', end='')

    if elmo_model == ELMoModel.Original:
        total_embedding_dim = Constants.ORIGINAL_ELMO_EMBEDDING_DIMENSION + glove_vectors_dim
    elif elmo_model == ELMoModel.Small:
        total_embedding_dim = Constants.SMALL_ELMO_EMBEDDING_DIMENSION + glove_vectors_dim


    hyperpartisan_model = JointModel(embedding_dim=total_embedding_dim,
                                     hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                                     num_layers=Constants.DEFAULT_NUM_LAYERS,
                                     sent_encoder_dropout_rate=Constants.DEFAULT_SENTENCE_ENCODER_DROPOUT_RATE,
                                     doc_encoder_dropout_rate=Constants.DEFAULT_DOCUMENT_ENCODER_DROPOUT_RATE,
                                     output_dropout_rate=Constants.DEFAULT_OUTPUT_ENCODER_DROPOUT_RATE,
                                     device=device,
                                     skip_connection=Constants.DEFAULT_SKIP_CONNECTION).to(device)

    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                             num_layers=Constants.DEFAULT_NUM_LAYERS,
                             sent_encoder_dropout_rate=Constants.DEFAULT_SENTENCE_ENCODER_DROPOUT_RATE,
                             doc_encoder_dropout_rate=Constants.DEFAULT_DOCUMENT_ENCODER_DROPOUT_RATE,
                             output_dropout_rate=Constants.DEFAULT_OUTPUT_ENCODER_DROPOUT_RATE,
                             device=device,
                             skip_connection=Constants.DEFAULT_SKIP_CONNECTION).to(device)
    
    load_model_state(hyperpartisan_model, hyperpartisan_model_checkpoint_path)
    load_model_state(joint_model, joint_model_checkpoint_path)

    print('Loading model state...Done')

    return hyperpartisan_model, joint_model

def create_contingency_table(targets, predictions1, predictions2):
    assert len(targets) == len(predictions1)
    assert len(targets) == len(predictions2)

    contingency_table = np.zeros((2, 2))

    targets_length = len(targets)
    contingency_table[0, 0] = sum([targets[i] == predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # both predictions are correct
    contingency_table[0, 1] = sum([targets[i] == predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # predictions1 is correct and predictions2 is wrong
    contingency_table[1, 0] = sum([targets[i] != predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # predictions1 is wrong and predictions2 is correct
    contingency_table[1, 1] = sum([targets[i] != predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # both predictions are wrong

    print(contingency_table)

    return contingency_table

def forward_full_hyperpartisan(
        joint_model: JointModel,
        dataloader: DataLoader,
        device: torch.device):

    all_targets = []
    all_predictions = []

    running_accuracy = 0

    total_length = len(dataloader)
    for step, hyperpartisan_data in enumerate(dataloader):
        print(f'Step {step+1}/{total_length}                  \r', end='')

        batch_inputs = hyperpartisan_data[0].to(device)
        batch_targets = hyperpartisan_data[1].to(device)
        batch_recover_idx = hyperpartisan_data[2].to(device)
        batch_num_sent = hyperpartisan_data[3].to(device)
        batch_sent_lengths = hyperpartisan_data[4].to(device)
        batch_feat = hyperpartisan_data[5].to(device)

        batch_predictions = joint_model.forward(batch_inputs, (batch_recover_idx,
                                                        batch_num_sent, batch_sent_lengths, batch_feat), task=TrainingMode.Hyperpartisan)

        accuracy = utils_helper.calculate_accuracy(batch_predictions, batch_targets)

        running_accuracy += accuracy
        all_targets += batch_targets.long().tolist()
        all_predictions += batch_predictions.round().long().tolist()

    final_accuracy = running_accuracy / (step + 1)

    return final_accuracy, all_targets, all_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperpartisan_model_checkpoint', type=str, required=True,
                        help='Path to load the hyperpartisan model')
    parser.add_argument('--joint_model_checkpoint', type=str, required=True,
                        help='Path to load the joint model')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
    parser.add_argument('--vector_cache_dir', type=str, default=Constants.DEFAULT_VECTOR_CACHE_DIR,
                        help='Directory where vectors would be cached')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str, required=True,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--hyperpartisan_batch_size', type=int, default=Constants.DEFAULT_HYPERPARTISAN_BATCH_SIZE,
                        help='Batch size for training on the hyperpartisan dataset')
    parser.add_argument('--metaphor_batch_size', type=int, default=Constants.DEFAULT_METAPHOR_BATCH_SIZE,
                        help='Batch size for training on the metaphor dataset')
    parser.add_argument('--deterministic', type=int, required=True,
                        help='The seed to be used when running deterministically. If nothing is passed, the program run will be stochastic')
    parser.add_argument('--alpha_value', type=int, default=0.05,
                        help='The alpha value which will be used to statistically test the significant difference')
    parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
                        help='ELMo model from which vectors are used')

    config = parser.parse_args()

    utils_helper.initialize_deterministic_mode(config.deterministic)

    device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")

    glove_vectors = utils_helper.load_glove_vectors(
        config.vector_file_name, config.vector_cache_dir, config.glove_size)

    hyperpartisan_model, joint_model = initialize_models(config.hyperpartisan_model_checkpoint,
                                                         config.joint_model_checkpoint,
                                                         device,
                                                         config.elmo_model,
                                                         glove_vectors.dim)
    
    _, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        glove_vectors=glove_vectors,
        elmo_model=config.elmo_model,
        lowercase_sentences=False,
        articles_max_length=Constants.DEFAULT_HYPERPARTISAN_MAX_LENGTH,
        load_train=False)

    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        batch_size=config.hyperpartisan_batch_size,
        shuffle=False)

    hyperpartisan_valid_accuracy, hyperpartisan_valid_targets, hyperpartisan_valid_predictions = forward_full_hyperpartisan(
        joint_model=hyperpartisan_model,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    hyperpartisan_f1, hyperpartisan_precision, hyperpartisan_recall = utils_helper.calculate_metrics(hyperpartisan_valid_targets, hyperpartisan_valid_predictions)

    print(f'Hyperpartisan F1 score: {hyperpartisan_f1}, Precision: {hyperpartisan_precision}, Recall: {hyperpartisan_recall}, Accuracy: {hyperpartisan_valid_accuracy}')

    joint_valid_accuracy, joint_valid_targets, joint_valid_predictions = forward_full_hyperpartisan(
        joint_model=joint_model,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    joint_f1, joint_precision, joint_recall = utils_helper.calculate_metrics(joint_valid_targets, joint_valid_predictions)

    print(f'Joint F1 score: {joint_f1}, Precision: {joint_precision}, Recall: {joint_recall}, Accuracy: {joint_valid_accuracy}')


    print('Calculating p value using McNemar\'s test...\r', end='')

    # roll your own t-test:
    N=10
    hyperpartisan_valid_predictions = np.array(hyperpartisan_valid_predictions)
    joint_valid_predictions = np.array(joint_valid_predictions)
    var_a = hyperpartisan_valid_predictions.var(ddof=1) # unbiased estimator, divide by N-1 instead of N
    var_b = joint_valid_predictions.var(ddof=1)
    s = np.sqrt( (var_a + var_b) / 2 ) # balanced standard deviation
    t = (hyperpartisan_valid_predictions.mean() - joint_valid_predictions.mean()) / (s * np.sqrt(2.0/N)) # t-statistic
    df = 2*N - 2 # degrees of freedom
    p = 1 - stats.t.cdf(np.abs(t), df=df) # one-sided test p-value
    print("t:\t", t, "p:\t", 2*p) # two-sided test p-value

    t2, p2 = ttest_ind(hyperpartisan_valid_predictions, joint_valid_predictions)
    print("t2:\t", t2, "p2:\t", p2)

    # contingency_table = create_contingency_table(
    #     hyperpartisan_valid_targets,
    #     hyperpartisan_valid_predictions,
    #     joint_valid_predictions)
    
    # result = mcnemar(contingency_table, exact=True)

    # p_value = permutation_test(hyperpartisan_valid_predictions, joint_valid_predictions,
    #                        method='approximate',
    #                        num_rounds=10000,
    #                        seed=0)

    # print('Calculating p value using McNemar\'s test...Done')

    # # summarize the finding
    # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
    # print(f'p-value: {p_value}')

    # # interpret the p-value
    # if p_value > config.alpha_value:
    #     print('Same proportions of errors (non-significant difference)')
    # else:
    #     print('Significant difference found')