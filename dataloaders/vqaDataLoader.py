import json
import os
from collections import Counter
from functools import lru_cache
import random

import regex as re
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

class VQADataset(Dataset):
    def __init__(
            self,
            dataDir,
            questionDataSubType,
            answersDataSubType,
            numCandidates,
            n_obs=None,
            outFile=None,
            append=True,
            versionType = "v2",
            taskType = "OpenEnded",
            dataType = "mscoco",
            candidateAnswerGenerator='most_common'
    ):
        super().__init__()

        self.dataDir = dataDir
        self.questionDataSubType = questionDataSubType
        self.answersDataSubType = answersDataSubType
        self.numCandidates = numCandidates
        self.versionType = versionType
        self.taskType = taskType
        self.dataType = dataType
        self.n_obs = n_obs
        self.candidateAnswerGenerator = candidateAnswerGenerator

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap    = { 'none': '0',
                              'zero': '0',
                              'one': '1',
                              'two': '2',
                              'three': '3',
                              'four': '4',
                              'five': '5',
                              'six': '6',
                              'seven': '7',
                              'eight': '8',
                              'nine': '9',
                              'ten': '10'
                              }
        self.articles     = ['a',
                             'an',
                             'the'
                             ]

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
                             '(', ')', '=', '+', '\\', '_', '-',
                             '>', '<', '@', '`', ',', '?', '!']

        self.allAnswers, self.qiPairs = self.getImageTextPairDataset()
        if(append and outFile is not None and os.path.exists(outFile)):
            results = json.load(open(outFile))
            qidsProcessed = results.keys()
        else:
            qidsProcessed = []
        self.qidsProcessed = [int(q) for q in qidsProcessed]
        self.qid = list(set(self.qiPairs.keys()))
        print("Out of {}/{} (Questions, Image) pairs already processed.".format( len(self.qidsProcessed), len(self.qiPairs.keys())))

    def getImageTextPairDataset(self):
        allAnswers, questionIdToAnswerMap = self.getAllAnswers()
        qiPairs = self.getQIPairs(questionIdToAnswerMap)
        return allAnswers, qiPairs

    def getQIPairs(self, questionIdToAnswerMap):
        quesFile = self.__getQuestionJson(dataSubType = self.questionDataSubType, versionType = self.versionType, taskType = self.taskType, dataType = self.dataType)
        qiPairs = {}
        imageRootDir = self.__getImageDir(dataType = self.dataType, dataSubType = self.questionDataSubType)
        if(self.dataType == 'mscoco'):
            PREFIX = "COCO"
        else:
            raise Exception("Prefix not defined for image files with dataType={}".format(self.dataType))

        if self.n_obs is None:
            self.n_obs = len(quesFile['questions'])
        for quesObj in quesFile['questions'][:self.n_obs]:
            qid = quesObj['question_id']
            temp = {}
            image_name = "{}_{}_{}.jpg".format(PREFIX, self.questionDataSubType, str(quesObj['image_id']).rjust(12, '0'))
            temp['image_path'] = os.path.join(imageRootDir, image_name)
            temp['question'] = quesObj['question']
            temp['answer'] = questionIdToAnswerMap[qid]
            qiPairs[qid] = temp
        return qiPairs

    @lru_cache()
    def getAllAnswers(self):
        annFile = self.__getAnnotationJson(dataSubType = self.answersDataSubType, versionType = self.versionType, dataType = self.dataType)
        answerCounts = {}
        questionIdToAnswerMap = {}
        if self.n_obs is None:
            self.n_obs = len(annFile['annotations'])
        for ann in tqdm(annFile['annotations'][:self.n_obs], desc = "Generating All Answers"):
            answer = self.__preProcessAnswer(ann['multiple_choice_answer'])
            answerCounts[answer] = answerCounts.get(answer, 0) + 1
            questionIdToAnswerMap[ann['question_id']] = answer
        return answerCounts, questionIdToAnswerMap

    def __getAnnotationJson(self, versionType, dataType, dataSubType):
        annFile ='%s/Annotations/%s_%s_%s_annotations.json'%(self.dataDir, versionType, dataType, dataSubType)
        return json.load(open(annFile))

    def __getQuestionJson(self, versionType, dataType, taskType, dataSubType):
        quesFile ='%s/Questions/%s_%s_%s_%s_questions.json'%(self.dataDir, versionType, taskType, dataType, dataSubType)
        return json.load(open(quesFile))

    def __getImageDir(self, dataType, dataSubType):
        return '%s/Images/%s/%s/' %(self.dataDir, dataType, dataSubType)

    def __processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def __processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def __preProcessAnswer(self, inText):
        outText = inText
        outText = outText.replace('\n', ' ')
        outText = outText.replace('\t', ' ')
        outText = outText.strip()
        outText = self.__processPunctuation(outText)
        outText = self.__processDigitArticle(outText)
        return outText

    def __len__(self):
        return len(self.qid)

    def getLabelsFromAllPossibleAnswers(self, question, correctAnswer, allAnswers, numCandidates):
        candidateAnswers = self.getCandidateAnswers(allAnswers, numCandidates)
        if(type(candidateAnswers) == dict):
            return self.labelCrossCandidateAnswers(question, correctAnswer, list(candidateAnswers.keys()))
        elif(type(candidateAnswers) == list):
            return self.labelCrossCandidateAnswers(question, correctAnswer, candidateAnswers)
        else:
            raise Exception("Class function \'getCandidateAnswers\' returns type {}, expected type is {} or {}".format(type(candidateAnswers), list, dict))

    def shuffle_lists(self, list1, list2, list3):
        indices = list(range(len(list1)))
        random.shuffle(indices)
        shuffled_list1 = []
        shuffled_list2 = []
        shuffled_list3 = []
        for i in indices:
            shuffled_list1.append(list1[i])
            shuffled_list2.append(list2[i])
            shuffled_list3.append(list3[i])
        return shuffled_list1, shuffled_list2, shuffled_list3

    def labelCrossCandidateAnswers(self, question, correctAnswer, candidateAnswers):
        labels = []
        questions = []
        candidateAnswers.append(correctAnswer)
        for candidateAnswer in candidateAnswers:
            questions.append(question)
            if candidateAnswer == correctAnswer:
                labels.append(1)
            else:
                labels.append(0)
        return self.shuffle_lists(questions, candidateAnswers, labels)

    def getCandidateAnswers(self, allAnswers, numCandidates):
        if self.candidateAnswerGenerator == "most_common":
            return self.getMostCommonCandidateAnswers(allAnswers, numCandidates)
        else:
            raise Exception("Available candidateAnswerGenerator are: \'most_common\', {} not configured".format(
                self.candidateAnswerGenerator))

    def getMostCommonCandidateAnswers(self, allAnswers, numCandidates):
        if type(allAnswers) == list:  # Assume equal prior probabilities
            return random.sample(allAnswers, numCandidates-1)
        elif type(allAnswers) == dict:  # Dictionary of prior probabilities
            return dict(Counter(allAnswers).most_common(numCandidates-1))
        else:
            raise Exception(
                "\'allAnswers\' is expected to be a {} of {} or a {} with prior probabilities but " +
                "found {}".format(list, str, dict, type(allAnswers)))

    def __getitem__(self, index):
        qid = self.qid[index]
        question = self.qiPairs[qid]['question']
        correctAnswer = self.qiPairs[qid]['answer']
        questions, answers, labels = self.getLabelsFromAllPossibleAnswers(question, correctAnswer, self.allAnswers, self.numCandidates)
        image = self.qiPairs[qid]['image_path']
        labels = torch.as_tensor(labels)

        return {"qid": qid, "image":image, "question":questions, "answers":answers, "labels":labels}

    def collate_fn(self, batch):
        qids = []
        images = []
        question = []
        answers = []
        labels = []
        for point in batch:
            qids.append(point['qid'])
            images.append(point['image'])
            question.append(point['question'])
            answers.append(point['answers'])
            labels.append(point['labels'])
        batch_encoding = {}
        batch_encoding['qids'] = qids
        batch_encoding['images'] = images
        batch_encoding['question'] = question
        batch_encoding['answers'] = answers
        batch_encoding['labels'] = torch.stack(labels)
        return batch_encoding