from tqdm import tqdm
import numpy as np
import json
import torch
import os
import pickle as pkl

class ImageTextPairDataset(torch.utils.data.Dataset):
	def __init__(self, allAnswers, qiPairs, getTextFromAllPossibleAnswers, numCandidates, qidsProcessed = []):
		self.allAnswers = allAnswers
		self.qiPairs = qiPairs
		self.getTextFromAllPossibleAnswers = getTextFromAllPossibleAnswers
		self.numCandidates = numCandidates
		self.qidsProcessed = [int(q) for q in qidsProcessed]
		self.qid = list(set(self.qiPairs.keys()) - set(self.qidsProcessed))
		print("Out of {}/{} (Questions, Image) pairs already processed.".format( len(self.qidsProcessed), len(self.qiPairs.keys())))
	def __getitem__(self, index):
		qid = self.qid[index]
		question = self.qiPairs[qid]['question']
		texts, answers = self.getTextFromAllPossibleAnswers(question, self.allAnswers, self.numCandidates)
		image = self.qiPairs[qid]['image_path']
		return {"qid": qid, "image":image, "texts":texts, "answers":answers}

	def __len__(self):
		return len(self.qid)

class CLIPVQA:
	def __init__(self, clipInterface, languageModel, vqaInterface):
		self.clipInterface = clipInterface
		self.languageModel = languageModel
		self.vqaInterface = vqaInterface

	def _collate_custom(self, batch):
		qids = []
		images = []
		texts = []
		answers = []
		for point in batch:
			qids.append(point['qid'])
			images.append(point['image'])
			texts.append(point['texts'])
			answers.append(point['answers'])
		return qids, images, texts, answers

	def generateImageTextPairs(self, evalDataSubType, answersDataSubType, numCandidates):
		allAnswers = self.vqaInterface.getAllAnswers(answersDataSubType)
		qiPairs = self.vqaInterface.getQIPairs(evalDataSubType)
		imageTextPairs = [] #TODO: Replace the list with data generator
		for qid in tqdm(qiPairs, desc = "Generating ImageText Pair"):
			question = qiPairs[qid]['question']
			texts, answers = self.languageModel.getTextFromAllPossibleAnswers(question, allAnswers, numCandidates)
			image = qiPairs[qid]['image_path']
			imageTextPairs.append((qid, image, texts, answers))
		return imageTextPairs

	def generateImageTextPairsWithSeparateAnswers(self, evalDataSubType, answersDataSubType, numCandidates):
		allAnswers = self.vqaInterface.getAllAnswers(answersDataSubType)
		qiPairs = self.vqaInterface.getQIPairs(evalDataSubType)
		imageTextPairs = [] #TODO: Replace the list with data generator
		for qid in tqdm(qiPairs, desc = "Generating ImageText Pair"):
			question = qiPairs[qid]['question']
			q, a, answers = self.languageModel.getTextFromAllPossibleAnswersWithSeparateAnswers(question, allAnswers, numCandidates)
			image = qiPairs[qid]['image_path']
			imageTextPairs.append((qid, image, q, a, answers))
		return imageTextPairs

	def getImageTextPairDataset(self, evalDataSubType, answersDataSubType, numCandidates, qidsProcessed = []):
		allAnswers = self.vqaInterface.getAllAnswers(answersDataSubType)
		qiPairs = self.vqaInterface.getQIPairs(evalDataSubType)
		dataSet = ImageTextPairDataset(allAnswers, qiPairs, self.languageModel.getTextFromAllPossibleAnswers, numCandidates, qidsProcessed)
		return dataSet

	def generateResults(self, evalDataSubType, answersDataSubType, numCandidates, outFile = None, experiment='concat_qa'):
		results = []
		if experiment == 'concat_qa':
			imageTextPairs = self.generateImageTextPairs(evalDataSubType, answersDataSubType, numCandidates)
			for (qid, image, texts, answers) in tqdm(imageTextPairs, desc = "Result Generation"):
				probs = self.clipInterface.getProbs(image, texts)
				predAnswer = answers[np.argmax(probs)]
				results.append({"answer":predAnswer, "question_id":qid})
		else:
			imageTextPairs = self.generateImageTextPairsWithSeparateAnswers(evalDataSubType, answersDataSubType, numCandidates)
			for (qid, image, q, a, answers) in tqdm(imageTextPairs, desc = "Result Generation"):
				probs = self.clipInterface.getProbsForSeparateAnswers(image, q, a)
				predAnswer = answers[np.argmax(probs)]
				results.append({"answer":predAnswer, "question_id":qid})
		if(outFile):
			json.dump(results, open(outFile, 'w'))
		return results

	def generateResultsDataLoader(self, evalDataSubType, answersDataSubType, numCandidates, pklImageFeaturesFile = None, outFile = None, append = True, oneTimeRunCount = None):
		if(append and outFile is not None and os.path.exists(outFile)):
			results = json.load(open(outFile))
			dataset = self.getImageTextPairDataset(evalDataSubType, answersDataSubType, numCandidates, qidsProcessed = results.keys())
		else:
			results = {}
			dataset = self.getImageTextPairDataset(evalDataSubType, answersDataSubType, numCandidates)
		
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn = self._collate_custom)
		if(pklImageFeaturesFile):
			preComputedFeatures = pkl.load(open(pklImageFeaturesFile, 'rb'))
		else:
			preComputedFeatures = None
		for batchIdx, (qids, images, texts, answers)  in enumerate(tqdm(dataloader, desc = "Result Generation", initial = len(results))):
			batch_size = len(qids)
			if(not oneTimeRunCount is None and batchIdx >= oneTimeRunCount):
				print("{} instances processed, breaking now".format(str(batchIdx)))
				break
			
			probs = self.clipInterface.getProbs(images, texts, preComputedImageFeatures = preComputedFeatures, batch_size=batch_size)
			for b in range(batch_size):
				prob = probs[b]
				answer = answers[b]
				predAnswer = answer[np.argmax(prob)]
				qid = qids[b]
				results[qid] = {"answer":predAnswer, "question_id":qid}
			if(batchIdx%100 == 0 and outFile):
				os.makedirs(os.path.dirname(outFile), exist_ok=True)
				json.dump(results, open(outFile, 'w'))
		if(outFile):
			os.makedirs(os.path.dirname(outFile), exist_ok=True)
			json.dump(results, open(outFile, 'w'))
		return results