from collections import Counter
import json
import string
import argparse

def load_data(prediction_file, testing_file):

    with open(prediction_file, 'r') as f1:
        predictoion_data = json.load(f1)
    with open(testing_file, 'r', encoding="utf-8") as f2:
        testing_data = json.load(f2)


    testing_pairs = []
    print(predictoion_data['101-1'])
    for pred_key in predictoion_data.keys():
        for test in testing_data['data']:

            test_key = test['id']
            if pred_key == test_key:
                print(pred_key)
                print(predictoion_data[pred_key])
                print(test['answers']['text'])
                testing_pairs.append({"key":pred_key, "predict": predictoion_data[pred_key], "trueLabel":test['answers']['text']})


    return testing_pairs



def string_clean(dirty_str):
    out_segs = []

    for char in dirty_str:
        if char in sp_char or char in string.punctuation:
            continue
        else:
            out_segs.append(char)
    clean_str = ''.join(out_segs)
    return clean_str

def f1_scoring(label, pred):
    prediction_tokens = Counter(pred)
    ground_truth_tokens = Counter(label)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        f1 = 0
    else:
        precision = 1.0 * num_same / len(pred)
        recall = 1.0 * num_same / len(label)
        f1 = (2 * precision * recall) / (precision + recall)
    return f1

# 標點符號
sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']

# 設定 arguments
parser = argparse.ArgumentParser()
parser.add_argument("--predictFile", type=str,
                    help="prediction json file for evaluation."
                    )
parser.add_argument("--testingFile", type=str,
                    help="testing json file for evaluation."
                    )

testing_pairs = load_data(parser.parse_args().predictFile, parser.parse_args().testingFile)


f1_list = []
for test_pair in testing_pairs:

    true_answer = test_pair['trueLabel']
    predictions = test_pair['predict']
    
    f1_score = 0
    for true in true_answer:
        # 清 label
        true = string_clean(true)

        # 清 pred
        predictions = string_clean(predictions)

        # f1 score
        f1_tmp =  f1_scoring(true, predictions)
        if f1_score < f1_tmp:
            f1_score = f1_tmp
            
    if f1_score > 1:
        print(f'{true_answer}, {predictions}')
    print(f1_score)
    f1_list.append(f1_score)

print(f'F1 :{sum(f1_list) / len(f1_list)}')

