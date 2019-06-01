import numpy as np
import os
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import permutation_test
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
import xml.sax

# PATHS
single_predictions_path = ""
joint_predictions_path = ""
inputDataset_path = ""

targets = []
class HyperpartisanNewsGroundTruthHandler(xml.sax.ContentHandler):
	def __init__(self):
		xml.sax.ContentHandler.__init__(self)

	def startElement(self, name, attrs):
		if name == "article":
			targets.append(int(attrs.getValue("hyperpartisan") == "true"))

def create_contingency_table(targets, predictions1, predictions2):
	assert len(targets) == len(predictions1)
	assert len(targets) == len(predictions2)

	contingency_table = np.zeros((2, 2))

	targets_length = len(targets)
	contingency_table[0, 0] = sum([targets[i] == predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # both predictions are correct
	contingency_table[0, 1] = sum([targets[i] == predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # predictions1 is correct and predictions2 is wrong
	contingency_table[1, 0] = sum([targets[i] != predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # predictions1 is wrong and predictions2 is correct
	contingency_table[1, 1] = sum([targets[i] != predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # both predictions are wrong

	return contingency_table

def calculate_ttest(single_predictions, joint_predictions):
	_, p_value = ttest_ind(single_predictions, joint_predictions)
	return p_value

def calculate_mcnemars_test(targets, single_predictions, joint_predictions):
	contingency_table = create_contingency_table(targets, single_predictions, joint_predictions)
	result = mcnemar(contingency_table, exact=True)
	return result.pvalue

def calculate_permutation_test(single_predictions, joint_predictions):
	p_value = permutation_test(single_predictions, joint_predictions, method='approximate', num_rounds=10000, seed=0)
	return p_value   


if not os.path.isfile(inputDataset_path):
	inputDataset_path = inputDataset_path + "/" + os.listdir(inputDataset_path)[0]

single_predictions = []
joint_predictions = []

with open(single_predictions_path, "r") as f:
	for line in f.readlines():
		single_predictions.append(int(line.split(" ")[1][:-1] == "True"))

with open(joint_predictions_path, "r") as f:
	for line in f.readlines():
		joint_predictions.append(int(line.split(" ")[1][:-1] == "True"))

with open(inputDataset_path) as f:
	xml.sax.parse(f, HyperpartisanNewsGroundTruthHandler())

p_value_ttest = calculate_ttest(single_predictions, joint_predictions)
p_value_mcnemars = calculate_mcnemars_test(targets, single_predictions, joint_predictions)
p_value_perm = calculate_permutation_test(single_predictions, joint_predictions)

print("ttest p-value: {:.6f}".format(p_value_ttest))
print("mcnemars test p-value: {:.6f}".format(p_value_mcnemars))
print("permuatiion test p-value: {:.6f}".format(p_value_perm))