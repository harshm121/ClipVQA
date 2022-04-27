import json

answers = "./data/Annotations/v2_mscoco_val2014_annotations.json"
final_answers = json.load(open(answers))['annotations']

t = {}
for i in final_answers:
    t[i["question_id"]] = {}
    t[i["question_id"]]["answer"] = i['multiple_choice_answer']
    t[i["question_id"]]["question_id"] = i['question_id']

f = "groundTruthDump.json"
json.dump(t, open("groundTruthDump.json", "w"))
