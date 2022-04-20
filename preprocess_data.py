import json
import os
import shutil

questions = json.load(open('val_color/v2_OpenEnded_mscoco_val2014_questions.json', "rb"))['questions']
annotations = json.load(open('val_color/v2_mscoco_val2014_annotations.json', "rb"))['annotations']
image_dir = 'data/Images/mscoco/val2014'
output_image_dir = 'outputData/Images/mscoco/val2014'
output_annotations = 'outputData/Annotations/v2_mscoco_val2014_annotations.json'
output_questions = 'outputData/Questions/v2_OpenEnded_mscoco_val2014_questions.json'

imageIdtoImageFile = {}
for curr_file in os.listdir(image_dir):
    id = curr_file[-10:-4]
    imageIdtoImageFile[id] = curr_file


for q in range(20768):
    image = questions[q]['image_id']
    while len(str(image)) != 6:
        image = '0' + str(image)
    curr_file = image_dir + '/' + imageIdtoImageFile[str(image)]
    output_file = output_image_dir + '/' + imageIdtoImageFile[str(image)]
    shutil.copyfile(curr_file, output_file)

questions = questions[0:20768]
annotations = annotations[0:20768]

questionList = json.load(open('val_color/v2_OpenEnded_mscoco_val2014_questions.json', "rb"))
questionList['questions'] = questions
annotationList = json.load(open('val_color/v2_mscoco_val2014_annotations.json', "rb"))
annotationList['annotations'] = annotations

json.dump(questionList, open(output_questions, "w"))
json.dump(annotationList, open(output_annotations, "w"))




