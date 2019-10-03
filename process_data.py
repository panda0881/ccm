import ujson as json
from tqdm import tqdm
# test = {"post": ["you", "mean", "the", "occupation", "that", "did", "happen", "?"], "response": ["no", "i", "mean", "the", "fighting", "invasion", "that", "the", "military", "made", "so", "many", "purple", "hearts", "for", "in", "anticipation", "for", "that", "we", "have", "n't", "used", "up", "to", "this", "day", "."]}
# f = open('resource.txt')
# data = json.load(f)
# f.close()


def convert_data(input_file_name, output_file_name, tmp_kg):
    print('input:', input_file_name, 'output:', output_file_name)
    with open(input_file_name, 'r') as f:
        test_data = json.load(f)
    post_triples = []
    all_triples = []
    all_entities = []

    all_examples_after_match = list()
    for tmp_example in tqdm(test_data):
        post = tmp_example['post']
        index = 0
        for word in post:
            try:
                entityIndex = tmp_kg['dict_csk_entities'][word]
                index += 1
                post_triples.append(index)
                all_triples.append(tmp_kg['postEntityToCSKTripleIndex'][word])
                all_entities.append(tmp_kg['postEntityToOtherCSKTripleEntities'][word])
            except:
                post_triples.append(0)

        tmp_example['post_triples'] = post_triples
        tmp_example['all_triples'] = all_triples
        tmp_example['all_entities'] = all_entities

        response_triples = []
        match_index = []
        match_triples = []
        for word in tmp_example['response']:
            try:
                found = False
                entityIndex = tmp_kg['dict_csk_entities'][word]
                for index, entitiesList in enumerate(tmp_example['all_entities']):
                    for subindex, entity in enumerate(entitiesList):
                        if (entity == entityIndex):
                            match_index.append([index + 1, subindex])
                            response_triples.append(tmp_example['all_triples'][index][subindex])
                            match_triples.append(tmp_example['all_triples'][index][subindex])
                            found = True
                            break
                if not found:
                    response_triples.append(-1)
                    match_index.append([-1, -1])
            except:
                response_triples.append(-1)
                match_index.append([-1, -1])

        tmp_example['response_triples'] = response_triples
        tmp_example['match_index'] = match_index
        tmp_example['match_triples'] = match_triples
        all_examples_after_match.append(tmp_example)
    with open(output_file_name, 'w') as f:
        for tmp_example in tqdm(all_examples_after_match):
            f.write(json.dumps(tmp_example))
            f.write('\n')


current_kg = dict()

current_kg['postEntityToCSKTripleIndex'] = {}
current_kg['postEntityToOtherCSKTripleEntities'] = {}
current_kg['dict_csk_triples'] = dict()
current_kg['csk_triples'] = dict()
index = 0
for triple in current_kg['csk_triples']:
    firstEntity = triple.split(',')[0]
    secondEntity = triple.split(',')[2].strip()
    if(not firstEntity in current_kg['postEntityToCSKTripleIndex']):
        current_kg['postEntityToCSKTripleIndex'][firstEntity] = []
    current_kg['postEntityToCSKTripleIndex'][firstEntity].append(index)
    if(not secondEntity in current_kg['postEntityToCSKTripleIndex']):
        current_kg['postEntityToCSKTripleIndex'][secondEntity] = []
    current_kg['postEntityToCSKTripleIndex'][secondEntity].append(index)

    if (not firstEntity in current_kg['postEntityToOtherCSKTripleEntities']):
        current_kg['postEntityToOtherCSKTripleEntities'][firstEntity] = []
    current_kg['postEntityToOtherCSKTripleEntities'][firstEntity].append(current_kg['dict_csk_entities'][secondEntity])
    if (not secondEntity in current_kg['postEntityToOtherCSKTripleEntities']):
        current_kg['postEntityToOtherCSKTripleEntities'][secondEntity] = []
    current_kg['postEntityToOtherCSKTripleEntities'][secondEntity].append(current_kg['dict_csk_entities'][firstEntity])
    index += 1

current_kg['indexToCSKTriple'] = {v: k for k,v in current_kg['dict_csk_triples'].items()}


# print(str(test))

convert_data('dialog_dataset/formatted_train.json', 'data/none/trainset.txt', current_kg)
convert_data('dialog_dataset/formatted_dev.json', 'data/none/validset.txt', current_kg)
convert_data('dialog_dataset/formatted_test.json', 'data/none/testset.txt', current_kg)
with open('data/none/resource.txt', 'w') as f:
    f.write(json.dumps(current_kg))
    f.write('\n')