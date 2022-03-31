import random
from LanguageModels.base import LanguageModelBase
from collections import Counter


class AppendWithPrefixSuffixModel(LanguageModelBase):
    def __init__(self, question_pre=' ', question_suf = ' ', answer_pre = ' ', answer_suf = ' ', candidateAnswerGenerator='most_common'):
        self.question_pre = question_pre
        self.question_suf = question_suf
        self.answer_pre = answer_pre
        self.answer_suf = answer_suf
        self.candidateAnswerGenerator = candidateAnswerGenerator

    def getText(self, question, answer):
        return f'{self.question_pre}{question}{self.question_suf}{self.answer_pre}{answer}{self.answer_suf}'

    def getCandidateAnswers(self, question, allAnswers, numCandidates):
        if self.candidateAnswerGenerator == "most_common":
            return self.getMostCommonCandidateAnswers(question, allAnswers, numCandidates)
        else:
            raise Exception("Availabe candidateAnswerGenerator are: \'most_common\', {} not configured".format(
                self.candidateAnswerGenerator))

    def getMostCommonCandidateAnswers(self, question, allAnswers, numCandidates):
        if type(allAnswers) == list:  # Assume equal prior probabilities
            return random.sample(allAnswers, numCandidates)
        elif type(allAnswers) == dict:  # Dictionary of prior probabilities
            return dict(Counter(allAnswers).most_common(numCandidates))
        else:
            raise Exception(
                "\'allAnswers\' is expected to be a {} of {} or a {} with prior probabilities but found {}".format(list,
                                                                                                                   str,
                                                                                                                   dict,
                                                                                                                   type(
                                                                                                                       allAnswers)))
