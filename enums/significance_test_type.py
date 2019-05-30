from enum import Enum


class SignificanceTestType(Enum):
    McNemars = 'mcnemars'
    Permutation = 'permutation'
    TTest = 'ttest'

    def __str__(self):
        return self.value
