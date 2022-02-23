from LanguageModels.appendQAModel import AppendQAModel
from CLIPInterface.clipInterface import CLIPInterface
from VQAInterface.vqaInterface import VQAInterface
from CLIPVQA.clipvqa import CLIPVQA

import os

args = {"numCandidates": 1000}
resultsPath = "./output/results/append_numCandidates_{}_resultsVal.json".format(args["numCandidates"])
appendModel = AppendQAModel(separator=" ", candidateAnswerGenerator='most_common')
clipInterface = CLIPInterface(device="cuda")
vqaInterface = VQAInterface(dataDir='./data', versionType="v2", taskType="OpenEnded", dataType="mscoco")

clipVqaModel = CLIPVQA(clipInterface, appendModel, vqaInterface)
images = ['./data/Images/mscoco/val2014/'+s for s in os.listdir('./data/Images/mscoco/val2014')]
allFeatures = clipInterface.getNormalisedImageFeatures(imageFilePath=images, batch_size=128, pklFilePath='./output/intermediate/normalisedFeatures.pkl')

results = clipVqaModel.generateResultsDataLoader(evalDataSubType="val2014", answersDataSubType="train2014",
                                                 numCandidates=args["numCandidates"], \
                                                 pklImageFeaturesFile = './output/intermediate/normalisedFeatures.pkl',
                                                 outFile=resultsPath)
