import itertools
import json
import math
import os
from argparse import ArgumentParser
import clip
import numpy as np

import pytorch_lightning as pl
import torch
from PIL import Image
import pickle as pkl
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.vqaDataLoader import VQADataset
from torch.nn import functional as F

class VQAModelClassifier(pl.LightningModule):
    def __init__(self,
                 hidden_dim,
                 lr,
                 scheduler_patience,
                 dataType = "mscoco",
                 **kwargs):
        super().__init__()
        # Automatically log all the arguments to the tensorboard
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        self.dataType = dataType
        self.save_hyperparameters()
        # model
        self.l1 = torch.nn.Linear(1024, self.hparams.hidden_dim)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 512)

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        self.dataset_class = VQADataset
        self.num_workers = self.hparams.num_workers
        model, preprocess = clip.load(self.hparams.modelType)
        for param in model.parameters():
            param.requires_grad = False
        self.clip = model
        self.image_preprocess = preprocess
        train_image_path = self.hparams.train_data_dir + '/Images/' + self.dataType + "/" + self.hparams.train_answersDataSubType
        self.train_images = [train_image_path+"/"+s for s in os.listdir(train_image_path)]
        val_image_path = self.hparams.val_data_dir + '/Images/' + self.dataType + "/" + self.hparams.val_answersDataSubType
        self.val_images = [val_image_path+"/"+s for s in os.listdir(val_image_path)]
        self.train_image_features = self.getNormalisedImageFeatures(
            imageFilePath=self.train_images, batch_size=128, pklFilePath=self.hparams.trainPklFilePath)
        self.val_image_features = self.getNormalisedImageFeatures(
            imageFilePath=self.val_images, batch_size=128, pklFilePath=self.hparams.valPklFilePath)
        results_path = {
            "train": self.hparams.resultsTrain,
            "val": self.hparams.resultsVal,
            "test": self.hparams.resultsVal,
        }
        self.resultsPathTrain = "./output/results/append_numCandidates_{}_{}.json".format(self.hparams.numCandidates, results_path["train"])
        self.resultsPathVal = "./output/results/append_numCandidates_{}_{}.json".format(self.hparams.numCandidates, results_path["val"])
        #self.resultsTrain = json.load(open(self.resultsPathTrain))
        self.resultsVal = {}


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
            image = self.image_preprocess(Image.open(tempBatch[0])).unsqueeze(0)
            for f in tempBatch[1:]:
                tempImage = self.image_preprocess(Image.open(f)).unsqueeze(0)
                image = torch.cat((image, tempImage), dim = 0)
            image = image.to("cuda")
            image_features = self.clip.encode_image(image)
            for idx, f in enumerate(tempBatch):
                allFeatures[f] = image_features[idx]
            done += len(tempBatch)
        pkl.dump(allFeatures, open(pklFilePath, 'wb'), protocol = 3)
        return allFeatures

    def getTextFeatures(self, questions, answers,
                                   batch_size = 1):
        '''
        NOTE: the number of text per image has to be same for all images! If there are different possible answers
        for each image, pad texts with some sort of <mask> token.
        '''
        # Getting embeddings for text
        all_question = list(itertools.chain.from_iterable(questions))
        question_per_image = len(all_question) // batch_size
        question = clip.tokenize(all_question).to("cuda")
        all_answer = list(itertools.chain.from_iterable(answers))
        answer_per_image = len(all_answer) // batch_size
        answer = clip.tokenize(all_answer).to("cuda")
        with torch.no_grad():
            question_features = self.clip.encode_text(question).to("cuda")
            question_features /= question_features.norm(dim=-1, keepdim=True)
            question_features = torch.stack(question_features.split(question_per_image), dim = 0) #batch_size x texts_per_image x 512
            answer_features = self.clip.encode_text(answer).to("cuda")
            answer_features /= answer_features.norm(dim=-1, keepdim=True)
            answer_features = torch.stack(answer_features.split(answer_per_image), dim = 0) #batch_size x texts_per_image x 512
        return question_features, answer_features

    def getImageFeatures(self, imageFilePath, preComputedImageFeatures = None):
        # Getting emebddings for images

        if(preComputedImageFeatures):

            if(type(imageFilePath) == str):
                image_features = torch.tensor(preComputedImageFeatures[imageFilePath]).unsqueeze(dim = 0)
            elif type(imageFilePath) == list and type(imageFilePath[0]) == str:
                image_features = []
                for f in imageFilePath:
                    image_features.append(torch.tensor(preComputedImageFeatures[f]))
                image_features = torch.stack(image_features, dim = 0) # batch_size x 512
                image_features = image_features.unsqueeze(dim = 1) # batch_size x 1 x 512
            else:
                raise Exception("imageFilePath type not known")
        else:
            if(type(imageFilePath) == str):
                image = self.image_preprocess(Image.open(imageFilePath)).unsqueeze(0).to("cuda")
            elif(type(imageFilePath) == list and type(imageFilePath[0]) == str):
                image = self.image_preprocess(Image.open(imageFilePath[0])).unsqueeze(0).to("cuda")
                for f in imageFilePath[1:]:
                    tempImage = self.image_preprocess(Image.open(f)).unsqueeze(0).to("cuda")
                    image = torch.cat((image, tempImage), dim = 0).to("cuda")
            else:
                raise Exception("imageFilePath is expected to be one of {} of {} or a single {} but found {}".format(list, str, str, type(imageFilePath)))
            with torch.no_grad():
                image_features = self.clip.encode_image(image).to("cuda")
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.unsqueeze(dim = 1) # batch_size x 1 x 512
        return image_features

    def getProbs(self, image_features, text_features):
        # Getting final probabilities
        logit_scale = self.clip.logit_scale.exp()
        # probs = batch_size x 1 x texts_per_image
        probs = (logit_scale * torch.matmul(image_features, text_features)).sigmoid()

        return probs

    def forward(self, text_features, image_features):
        text_features = text_features.float()
        image_features = image_features.float()
        text_features = self.l1(text_features)
        text_features = self.relu(text_features)
        text_features = self.l2(text_features)
        text_features = text_features.permute(0,2,1)
        probs = self.getProbs(image_features, text_features)
        return probs

    """def predict_step(self, batch, batch_idx):
        qids = batch['qids']
        images = batch['images']
        question = batch['question']
        answers = batch['answers']
        labels = batch['labels']

        question_features, answer_features = self.getTextFeatures(question, answers, len(qids))
        image_features = self.getImageFeatures(images, self.train_image_features)
        text_features = torch.cat((question_features, answer_features), 2).to("cuda")

        probs = self.forward(text_features, image_features)
        return probs"""

    def training_step(self, batch, batch_idx):
        qids = batch['qids']
        images = batch['images']
        question = batch['question']
        answers = batch['answers']
        labels = batch['labels'].float()

        question_features, answer_features = self.getTextFeatures(question, answers, len(qids))
        image_features = self.getImageFeatures(images, self.train_image_features)
        text_features = torch.cat((question_features, answer_features), 2).to("cuda")
        probs = self.forward(text_features, image_features)

        probs = probs.reshape(probs.shape[0], probs.shape[2])
        probs = probs.flatten()
        labels = labels.flatten()
        loss = F.binary_cross_entropy(probs, labels)
        self.log('trn_loss', loss.item()) # Automatic aggregation in the background
        return loss

    def validation_step(self, batch, batch_idx):
        qids = batch['qids']
        images = batch['images']
        question = batch['question']
        answers = batch['answers']
        labels = batch['labels'].float()

        question_features, answer_features = self.getTextFeatures(question, answers, len(qids))
        image_features = self.getImageFeatures(images, self.val_image_features)
        text_features = torch.cat((question_features, answer_features), 2).to("cuda")

        probs = self.forward(text_features, image_features)
        probs = probs.reshape(probs.shape[0], probs.shape[2])
        probs = probs.flatten()
        labels = labels.flatten()

        loss = F.binary_cross_entropy(probs, labels)
        self.log('val_loss', loss.item()) # Automatic aggregation in the background
        self.log('hp_metric', loss.item()) # Automatic aggregation in the background

        for b in range(len(qids)):
            prob = probs[b]
            answer = answers[b]
            predAnswer = answer[torch.argmax(prob)]
            qid = qids[b]
            self.resultsVal[qid] = {"answer":predAnswer, "question_id":qid}
        return loss

    def on_validation_end(self):
        print("===IAMWHATIAM==")
        os.makedirs(os.path.dirname(self.resultsPathVal), exist_ok=True)
        json.dump(self.resultsVal, open(self.resultsPathVal, 'w'))


    def test_step(self, batch, batch_idx):
        qids = batch['qids']
        images = batch['images']
        question = batch['question']
        answers = batch['answers']
        labels = batch['labels'].float()

        question_features, answer_features = self.getTextFeatures(question, answers, len(qids))
        image_features = self.getImageFeatures(images, self.val_image_features)
        text_features = torch.cat((question_features, answer_features), 2).to("cuda")

        probs = self.forward(text_features, image_features)
        probs = probs.reshape(probs.shape[0], probs.shape[2])

        loss = F.cross_entropy(labels, probs)
        self.log('tst_loss', loss.item()) # Automatic aggregation in the background
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_dataset(self, data_dir, questionDataSubType, answersDataSubType, n_obs, outFile=None) -> VQADataset:
        dataset = self.dataset_class(
            data_dir,
            questionDataSubType,
            answersDataSubType,
            self.hparams.numCandidates,
            n_obs,
            outFile=outFile
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        n_obs = self.n_obs[type_path]
        dataset = None
        if type_path=="train":
            dataset = self.get_dataset(self.hparams.train_data_dir,
                                       self.hparams.train_questionDataSubType,
                                       self.hparams.train_answersDataSubType,
                                       n_obs,
                                       outFile=self.resultsPathTrain)
        if type_path=="val":
            dataset = self.get_dataset(self.hparams.val_data_dir,
                                       self.hparams.val_questionDataSubType,
                                       self.hparams.val_answersDataSubType,
                                       n_obs,
                                       outFile=self.resultsPathTrain)
        if type_path=="test":
            dataset = self.get_dataset(self.hparams.test_data_dir,
                                       self.hparams.test_questionDataSubType,
                                       self.hparams.test_answersDataSubType,
                                       n_obs)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=None,
        )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader =  self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)
        return dataloader

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=1000)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--scheduler_patience', default=10, type=int)
        parser.add_argument('--modelType', default="ViT-B/32", type=str)
        return parser