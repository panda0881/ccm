import json


dialog_train = list()
with open('dialog_dataset/dialogues_train.txt', 'r') as f:
    for line in f:
        tmp_sentences = line.split('__eou__')[:-1]
        sentence_number = len(tmp_sentences)
        for i in range(len(tmp_sentences)-1):
            tmp_pair = dict()
            tmp_pair['post'] = tmp_sentences[i]
            tmp_pair['response'] = tmp_sentences[i+1]
            dialog_train.append(tmp_pair)

with open('dialog_dataset/formatted_train.json', 'w') as f:
    json.dump(dialog_train, f)

dialog_dev = list()
with open('dialog_dataset/dialogues_validation.txt', 'r') as f:
    for line in f:
        tmp_sentences = line.split('__eou__')[:-1]
        sentence_number = len(tmp_sentences)
        for i in range(len(tmp_sentences)-1):
            tmp_pair = dict()
            tmp_pair['post'] = tmp_sentences[i]
            tmp_pair['response'] = tmp_sentences[i+1]
            dialog_dev.append(tmp_pair)

with open('dialog_dataset/formatted_dev.json', 'w') as f:
    json.dump(dialog_dev, f)

dialog_test = list()
with open('dialog_dataset/dialogues_test.txt', 'r') as f:
    for line in f:
        tmp_sentences = line.split('__eou__')[:-1]
        sentence_number = len(tmp_sentences)
        for i in range(len(tmp_sentences)-1):
            tmp_pair = dict()
            tmp_pair['post'] = tmp_sentences[i]
            tmp_pair['response'] = tmp_sentences[i+1]
            dialog_test.append(tmp_pair)

with open('dialog_dataset/formatted_test.json', 'w') as f:
    json.dump(dialog_test, f)

