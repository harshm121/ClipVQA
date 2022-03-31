from LanguageModels.appendWithPrefixSuffixModel import AppendWithPrefixSuffixModel
from CLIPInterface.clipInterface import CLIPInterface
from VQAInterface.vqaInterface import VQAInterface
from CLIPVQA.clipvqa import CLIPVQA

import os, argparse
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Full Pipeline Training')

    parser.add_argument('--num-candidates', type=int, default=1000,
                        help='Shorter side transformation.')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_arguments()
    resultsPath = "./output/results/prefixSuffix_numCandidates_{}_resultsVal.json".format(args.num_candidates)
    appendModel = AppendWithPrefixSuffixModel(question_pre='Question: ', question_suf = '', answer_pre = ' Answer: ', answer_suf = '', candidateAnswerGenerator='most_common')
    clipInterface = CLIPInterface(device="cuda")
    vqaInterface = VQAInterface(dataDir='./data', versionType="v2", taskType="OpenEnded", dataType="mscoco")

    clipVqaModel = CLIPVQA(clipInterface, appendModel, vqaInterface)
    images = ['./data/Images/mscoco/val2014/'+s for s in os.listdir('./data/Images/mscoco/val2014')]
    allFeatures = clipInterface.getNormalisedImageFeatures(imageFilePath=images, batch_size=128, pklFilePath='./output/intermediate/normalisedFeatures.pkl')

    results = clipVqaModel.generateResultsDataLoader(evalDataSubType="val2014", answersDataSubType="train2014",
                                                    numCandidates = args.num_candidates, \
                                                    pklImageFeaturesFile = './output/intermediate/normalisedFeatures.pkl',
                                                    outFile=resultsPath)
