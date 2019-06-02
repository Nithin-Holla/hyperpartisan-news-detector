from model.JointModel import JointModel
from os import listdir
import torch
from enums.training_mode import TrainingMode


class Ensemble(object):

	def __init__(self,
				 path_to_models,
				 sent_encoder_hidden_dim,
				 doc_encoder_hidden_dim,
			 	 num_layers,
			 	 skip_connection,
			 	 include_article_features,
			 	 document_encoder_model,
			 	 pre_attention_layer,
			 	 total_embedding_dim,
				 device
				 ):

		self.models = []
		self.device = device

		checkpoint_paths = listdir(path_to_models)
		self.num_models = len(checkpoint_paths)

		for i, checkpoint_path in enumerate(checkpoint_paths):
			checkpoint = torch.load(path_to_models + checkpoint_path, map_location=device)

			joint_model = JointModel(embedding_dim=total_embedding_dim,
									 sent_encoder_hidden_dim=sent_encoder_hidden_dim,
									 doc_encoder_hidden_dim=doc_encoder_hidden_dim,
									 num_layers=num_layers,
									 sent_encoder_dropout_rate=.0,
									 doc_encoder_dropout_rate=.0,
									 output_dropout_rate=.0,
									 device=device,
									 skip_connection=skip_connection,
									 include_article_features=include_article_features,
									 doc_encoder_model=document_encoder_model,
									 pre_attn_layer=pre_attention_layer
									 ).to(device)
		
			joint_model.load_state_dict(checkpoint['model_state_dict'])

			self.models.append(joint_model)

	def forward(self, x, extra_args, task, return_attention=False):
		if task == TrainingMode.Hyperpartisan:

			predictions = []

			for i, model in enumerate(self.models):
				if return_attention:
					pred, word_attention, sentence_attention = model.forward(x, extra_args, task, return_attention=True)
					if i == 0:
						word_attentions = word_attention.unsqueeze(0)
						sentence_attentions = sentence_attention.unsqueeze(0)
					else:
						word_attentions = torch.cat((word_attentions, word_attention.unsqueeze(0)), dim=0)
						sentence_attentions = torch.cat((sentence_attentions, sentence_attention.unsqueeze(0)), dim=0)
				else:
					pred = model.forward(x, extra_args, task)
				
					predictions.append(pred[0])

			result_predictions = sum(predictions)/self.num_models
			if return_attention:
				return result_predictions, word_attentions.mean(dim=0).squeeze(), sentence_attentions.mean(dim=0).squeeze()
			else:
				return result_predictions

		elif task == TrainingMode.Metaphor:

			predictions = []
			for i, model in enumerate(self.models):

				if not i:
					predictions = model.forward(x, extra_args, task).squeeze()
				else:
					predictions += model.forward(x, extra_args, task).squeeze()

			mean_predictions = predictions / self.num_models

			return mean_predictions
				

	def eval(self):

		for model in self.models:
			model.eval()