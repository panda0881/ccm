import ujson as json


with open('kgs/lemmatized_commonsense_knowledge.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open('kgs/conceptnet.txt', 'w') as f:
    for r in test_data:
        for tmp_k in test_data[r]:
            f.write(r)
            f.write('\t')
            f.write(tmp_k['head'])
            f.write('\t')
            f.write(tmp_k['tail'])
            f.write('\n')


print('end')
