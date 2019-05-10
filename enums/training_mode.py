from enum import Enum

class TrainingMode(Enum):
    Metaphor = 'Metaphor'
    Hyperpartisan = 'Hyperpartisan'
    Joint = 'Joint'

    def __str__(self):
        return self.value