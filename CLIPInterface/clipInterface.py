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

	def getProbs(self, imageFilePath, texts, preComputedImageFeatures = None):
		# Getting embeddings for text
		text = clip.tokenize(texts).to(self.device)
		with torch.no_grad():
			text_features = self.clip.encode_text(text)
			text_features /= text_features.norm(dim=-1, keepdim=True)
			text_features.to(self.device)

		# Getting emebddings for images
		if(preComputedImageFeatures):
			if(type(imageFilePath) == str):
				image_features = torch.tensor(preComputedImageFeatures[imageFilePath]).to(self.device)
			elif(type(imageFilePath) == list and type(imageFilePath[0]) == str):
				image_features = torch.tensor(preComputedImageFeatures[imageFilePath[0]]).to(self.device)
				for f in imageFilePath[1:]:
					tempImage = torch.tensor(preComputedImageFeatures[f]).to(self.device)
					image_features = torch.cat((image_features, tempImage), dim = 0)
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

		# Getting final probabilities
		with torch.no_grad():
			logit_scale = self.clip.logit_scale.exp()
			probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1).detach().cpu().numpy()

		return probs

	def getNormalisedImageFeatures(self, imageFilePath, batch_size, pklFilePath):
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
			try:
				pkl.dump(allFeatures, open(pklFilePath, 'wb'), protocol = 3)
			except Exception as e:
				print(e)
		try:
			pkl.dump(allFeatures, open(pklFilePath, 'wb'), protocol = 3)
		except Exception as e:
			print(e)
		return allFeatures












	