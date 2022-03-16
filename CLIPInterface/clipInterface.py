import itertools
import torch
import clip
from PIL import Image
import pickle as pkl
import os
from tqdm import tqdm
import math

class CLIPInterface:
	def __init__(self, modelType = "ViT-B/32", device = "cuda" if torch.cuda.is_available() else "cpu"):
		self.device = device
		model, preprocess = clip.load(modelType, device=device)
		self.clip = model
		self.image_preprocess = preprocess

	def getProbs(self, imageFilePath, texts, preComputedImageFeatures = None, batch_size = 1):
		'''
		NOTE: the number of text per image has to be same for all images! If there are different possible answers
		for each image, pad texts with some sort of <mask> token. 
		'''
		# Getting embeddings for text
		all_texts = list(itertools.chain.from_iterable(texts))
		texts_per_image = len(all_texts) // batch_size
		text = clip.tokenize(all_texts).to(self.device)
		with torch.no_grad():
			text_features = self.clip.encode_text(text)
			text_features /= text_features.norm(dim=-1, keepdim=True)
			text_features.to(self.device)
			text_features = torch.stack(text_features.split(texts_per_image), dim = 0) #batch_size x texts_per_image x 512
			text_features = text_features.permute(0, 2, 1) #batch_size x 512 x texts_per_image
		# Getting emebddings for images
		
		if(preComputedImageFeatures):
		
			if(type(imageFilePath) == str):
				image_features = torch.tensor(preComputedImageFeatures[imageFilePath]).unsqueeze(dim = 0)
				image_features = image_features.to(self.device)
			elif type(imageFilePath) == list and type(imageFilePath[0]) == str:
				image_features = []
				for f in imageFilePath:
					image_features.append(torch.tensor(preComputedImageFeatures[f]).to(self.device))
				image_features = torch.stack(image_features, dim = 0) # batch_size x 512
				image_features = image_features.unsqueeze(dim = 1) # batch_size x 1 x 512
			else:
				raise Exception("imageFilePath type not known")
		else:
			if(type(imageFilePath) == str):
				image = self.image_preprocess(Image.open(imageFilePath)).unsqueeze(0).to(self.device)	
			elif(type(imageFilePath) == list and type(imageFilePath[0]) == str):
				image = self.image_preprocess(Image.open(imageFilePath[0])).unsqueeze(0).to(self.device)
				for f in imageFilePath[1:]:
					tempImage = self.image_preprocess(Image.open(f)).unsqueeze(0).to(self.device)
					image = torch.cat((image, tempImage), dim = 0)
			else:
				raise Exception("imageFilePath is expected to be one of {} of {} or a single {} but found {}".format(list, str, str, type(imageFilePath)))
			with torch.no_grad():
				image_features = self.clip.encode_image(image)
				image_features /= image_features.norm(dim=-1, keepdim=True)
				image_features = image_features.unsqueeze(dim = 1) # batch_size x 1 x 512
		# Getting final probabilities
		with torch.no_grad():
			logit_scale = self.clip.logit_scale.exp()
			# probs = batch_size x 1 x texts_per_image
			probs = (logit_scale * torch.matmul(image_features, text_features)).softmax(dim=-1).detach().cpu().numpy()

		return probs

	def getNormalisedImageFeatures(self, imageFilePath, batch_size, pklFilePath):
		os.makedirs(os.path.dirname(pklFilePath), exist_ok=True)
		batchesDone = 0
		if(os.path.exists(pklFilePath)):
			allFeatures = pkl.load(open(pklFilePath, 'rb'))
		else:
			allFeatures = {}

		if(type(imageFilePath) == str):
			imageFilePath = [imageFilePath]
		elif not(type(imageFilePath) == list and type(imageFilePath[0]) == str):
			raise Exception("imageFilePath is expected to be one of {} of {} or a single {} but found {}".format(list, str, str, type(imageFilePath)))

		imageFilePath = list(set(imageFilePath) - set(allFeatures.keys()))
		totalBatches = math.ceil(len(imageFilePath) / batch_size)
		done = 0
		for batchesDone in tqdm(range(totalBatches), desc = "Batches processed"):
			tempBatch = imageFilePath[done:done + batch_size]
			image = self.image_preprocess(Image.open(tempBatch[0])).unsqueeze(0).to(self.device)
			for f in tempBatch[1:]:
				tempImage = self.image_preprocess(Image.open(f)).unsqueeze(0).to(self.device)
				image = torch.cat((image, tempImage), dim = 0)
			image_features = self.clip.encode_image(image).detach().cpu().numpy()
			for idx, f in enumerate(tempBatch):
				allFeatures[f] = image_features[idx]
			done += len(tempBatch)
		pkl.dump(allFeatures, open(pklFilePath, 'wb'), protocol = 3)
		return allFeatures












	