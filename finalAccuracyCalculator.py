import json

file = "./output/results/append_numCandidates_500_resultsVal.json"
answers = "./data/Annotations/v2_mscoco_val2014_annotations.json"

contents = json.load(open(file))
final_answers = json.load(open(answers))['annotations']

ques2answer = {}
for i in final_answers:
    ques2answer[i['question_id']] = i['multiple_choice_answer']
    if i['question_id'] == 14549000:
        print('here')
totco = 0
co = 0
for key, value in contents.items():
    predicted_answer = value['answer']
    correct_answer = ques2answer[int(key)]
    totco+=1
    if predicted_answer == correct_answer:
        co+=1
print("Acc :" )
print(co/totco)


