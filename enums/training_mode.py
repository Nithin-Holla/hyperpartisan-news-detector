from enum import Enum


class TrainingMode(Enum):
    Metaphor = 'Metaphor'
    Hyperpartisan = 'Hyperpartisan'
    JointEpochs = 'JointEpochs'
    JointBatches = 'JointBatches'

    def __str__(self):
        return self.value

    @staticmethod
    def contains_metaphor(training_mode) -> bool:
        result = (training_mode == TrainingMode.Metaphor or training_mode ==
                  TrainingMode.JointEpochs)
        return result

    @staticmethod
    def contains_hyperpartisan(training_mode) -> bool:
        result = (training_mode == TrainingMode.Hyperpartisan or training_mode ==
                  TrainingMode.JointEpochs or training_mode == TrainingMode.JointBatches)
        return result
