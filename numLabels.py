import json

answers = "./data/Annotations/v2_mscoco_train2014_annotations.json"
final_answers = json.load(open(answers))['annotations'][:10000]

t = set()

for i in final_answers:
    t.add(i['multiple_choice_answer'])

print(len(t))
