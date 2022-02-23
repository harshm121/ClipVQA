import abc

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class LanguageModelBase(ABC):
    @property
    @abc.abstractmethod
    def getText(self, question, answer):
        pass

    """
    	question: the question as a string
    	allAnswers: list of all possible answers - prior probability is then considered as uniform across all asnwers
    				OR
    			  : dictionary with key as a possible answer and it's value as the prior probability
		k: number of candidates to return

    	returns: a list of possible answers or a dictionary with posterior probabilities.
    """
    @property
    @abc.abstractmethod
    def getCandidateAnswers(self, question, allAnswers, numCandidates):
    	pass


    """
    	question: the question as a string
    	candidateAnswers: candidateAnswers to be crossed with for the question
		
		returns: a list of text and the corresponding answers
    """
    def questionCrossCandidateAnswers(self, question, candidateAnswers):
    	qXa = []
    	for candidateAnswer in candidateAnswers:
    		qXa.append(self.getText(question, candidateAnswer))
    	return qXa, candidateAnswers

    """
    	question: the question as a string
    	allAnswers: list of all possible answers - prior probability is then considered as uniform across all asnwers
    				OR
    			  : dictionary with key as a possible answer and it's value as the prior probability
		k: number of candidates to return

    	returns: a list of text and the corresponding answers
    """
    def getTextFromAllPossibleAnswers(self, question, allAnswers, numCandidates):
    	candidateAnswers = self.getCandidateAnswers(question, allAnswers, numCandidates)
    	if(type(candidateAnswers) == dict):
    		return self.questionCrossCandidateAnswers(question, list(candidateAnswers.keys()))
    	elif(type(candidateAnswers) == list):
    		return self.questionCrossCandidateAnswers(question, candidateAnswers)
    	else:
    		raise Exception("Class function \'getCandidateAnswers\' returns type {}, expected type is {} or {}".format(type(candidateAnswers), list, dict))

