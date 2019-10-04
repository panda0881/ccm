import ujson as json
from tqdm import tqdm
from OpenKE.config import Config
from OpenKE.models.TransE import TransE
import tensorflow as tf
import numpy as np

# with open('kgs/lemmatized_commonsense_knowledge.json', 'r', encoding='utf-8') as f:
#     test_data = json.load(f)
#
# with open('kgs/conceptnet.txt', 'w', encoding='utf-8') as f:
#     for r in test_data:
#         for tmp_k in tqdm(test_data[r]):
#             # print(tmp_k)
#             f.write(r)
#             f.write('\t')
#             f.write(tmp_k['head'])
#             f.write('\t')
#             f.write(tmp_k['tail'])
#             f.write('\n')
#             # break


def train_TransE(target_folder):
    con = Config()
    # Input training files from benchmarks/FB15K/ folder.
    con.set_in_path(target_folder+'/')
    con.set_log_on(1)  # set to 1 to print the loss

    con.set_work_threads(30)
    con.set_train_times(10)
    con.set_nbatches(5)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(50)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(target_folder+"/model.vec.tf", steps=50)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(target_folder + "/embedding.vec.json")
    # Initialize experimental settings.
    con.init()
    # Set the knowledge embedding model
    # print(con.get_parameter_lists())
    con.set_model(TransE)
    # Train the model.
    # print(con.get_parameter_lists())
    # print(con.get_parameters())
    con.run()

    embeddings = con.get_parameters()
    # print(con.get_parameter_lists())
    print(con.model.ent_embeddings)
    print(con.model.rel_embeddings)


    # # we need to convert the embedding to txt
    # with open(target_folder + "/embedding.vec.json", "r") as f:
    #     dic = json.load(f)
    #
    print('dic:')
    print(embeddings)
    for key in embeddings:
        print(key)
    #
    # ent_embs, rel_embs = dic['ent_embeddings'], dic['rel_embeddings']
    #
    # with open(target_folder+'/entity_vector.json', 'w') as f:
    #     json.dump(ent_embs, f)
    #
    # with open(target_folder+'/relation_vector.json', 'w') as f:
    #     json.dump(rel_embs, f)


def prepare_kg(source_resource, target_folder):
    current_kg = dict()
    current_kg['csk_triples'] = list()
    current_kg['csk_entities'] = list()
    current_kg['csk_relations'] = list()
    all_vocab = list()

    with open('kgs/conceptnet.txt', 'r', encoding='utf-8') as f:
        print('We are working on kg:', 'kgs/conceptnet.txt')
        for line in f:
            tmp_words = line[:-1].split('\t')
            tmp_head = tmp_words[1]
            tmp_relation = tmp_words[0]
            tmp_tail = tmp_words[2]
            tmp_triplet = tmp_head+'$$'+tmp_relation+'$$'+tmp_tail
            current_kg['csk_triples'].append(tmp_triplet)
            current_kg['csk_entities'].append(tmp_head)
            current_kg['csk_entities'].append(tmp_tail)
            current_kg['csk_relations'].append(tmp_relation)
            for w in tmp_head.split(' '):
                all_vocab.append(w)
            for w in tmp_tail.split(' '):
                all_vocab.append(w)

    with open(source_resource, 'r', encoding='utf-8') as f:
        print('We are working on kg:', source_resource)
        for line in f:
            tmp_words = line[:-1].split('\t')
            tmp_head = tmp_words[1]
            tmp_relation = tmp_words[0]
            tmp_tail = tmp_words[2]
            tmp_triplet = tmp_head+'$$'+tmp_relation+'$$'+tmp_tail
            current_kg['csk_triples'].append(tmp_triplet)
            current_kg['csk_entities'].append(tmp_head)
            current_kg['csk_entities'].append(tmp_tail)
            current_kg['csk_relations'].append(tmp_relation)
            for w in tmp_head.split(' '):
                all_vocab.append(w)
            for w in tmp_tail.split(' '):
                all_vocab.append(w)

    all_vocab = list(set(all_vocab))
    current_kg['csk_triples'] = list(set(current_kg['csk_triples']))
    current_kg['csk_entities'] = list(set(current_kg['csk_entities']))
    current_kg['csk_relations'] = list(set(current_kg['csk_relations']))
    current_kg['vocab_dict'] = dict()
    current_kg['dict_csk'] = dict()
    current_kg['dict_csk_triples'] = dict()
    current_kg['dict_csk_relations'] = dict()

    for w in all_vocab:
        current_kg['vocab_dict'][w] = str(len(current_kg['vocab_dict']))

    for tmp_concept in current_kg['csk_entities']:
        current_kg['dict_csk'][tmp_concept] = str(len(current_kg['dict_csk']))

    for tmp_relation in current_kg['csk_relations']:
        current_kg['dict_csk_relations'][tmp_relation] = str(len(current_kg['dict_csk_relations']))

    for tmp_triplet in current_kg['csk_triples']:
        current_kg['dict_csk_triples'][tmp_triplet] = str(len(current_kg['dict_csk_triples']))

    with open(target_folder+'/current_kg.json', 'w', encoding='utf-8') as f:
        json.dump(current_kg, f)

    with open(target_folder+'/train2id.txt', 'w', encoding='utf-8') as f:
        print('Number of triplets:', len(current_kg['csk_triples']))
        f.write(str(len(current_kg['csk_triples'])))
        f.write('\n')
        for tmp_triplet in current_kg['csk_triples']:
            head_id = current_kg['dict_csk'][tmp_triplet.split('$$')[0]]
            tail_id = current_kg['dict_csk'][tmp_triplet.split('$$')[2]]
            relation_id = current_kg['dict_csk_relations'][tmp_triplet.split('$$')[1]]
            f.write(head_id)
            f.write('\t')
            f.write(tail_id)
            f.write('\t')
            f.write(relation_id)
            f.write('\n')
    with open(target_folder+'/entity2id.txt', 'w', encoding='utf-8') as f:
        print('Number of entity:', len(current_kg['csk_entities']))
        f.write(str(len(current_kg['csk_entities'])))
        f.write('\n')
        for tmp_entity in current_kg['csk_entities']:
            f.write(tmp_entity)
            f.write('\t')
            f.write(current_kg['dict_csk'][tmp_entity])
            f.write('\n')

    with open(target_folder+'/relation2id.txt', 'w', encoding='utf-8') as f:
        print('Number of relation:', len(current_kg['csk_relations']))
        f.write(str(len(current_kg['csk_relations'])))
        f.write('\n')
        for tmp_relation in current_kg['csk_relations']:
            f.write(tmp_relation)
            f.write('\t')
            f.write(current_kg['dict_csk_relations'][tmp_relation])
            f.write('\n')
    print('Finish preparing the kg')
    print('Start to train the TransE')
    train_TransE(target_folder)

def convert_data(input_file_name, output_file_name, tmp_kg):
    print('input:', input_file_name, 'output:', output_file_name)
    with open(input_file_name, 'r') as f:
        test_data = json.load(f)
    all_examples_after_match = list()
    for tmp_example in tqdm(test_data):
        new_example = dict()
        post = tmp_example['post'].split(' ')
        index = 0
        post_triples = []
        all_triples = []
        all_entities = []
        for word in post:
            try:
                entityIndex = tmp_kg['dict_csk_entities'][word]
                index += 1
                post_triples.append(index)
                all_triples.append(tmp_kg['postEntityToCSKTripleIndex'][word])
                all_entities.append(tmp_kg['postEntityToOtherCSKTripleEntities'][word])
            except:
                post_triples.append(0)

        new_example['post_triples'] = post_triples
        new_example['all_triples'] = all_triples
        new_example['all_entities'] = all_entities

        response_triples = []
        match_index = []
        match_triples = []
        for word in tmp_example['response'].split(' '):
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

        new_example['response_triples'] = response_triples
        new_example['match_index'] = match_index
        new_example['match_triples'] = match_triples
        new_example['post'] = tmp_example['post'].split(' ')
        new_example['response'] = tmp_example['response'].split(' ')
        all_examples_after_match.append(new_example)
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for tmp_example in tqdm(all_examples_after_match):
            # print(tmp_example)
            f.write(json.dumps(tmp_example))
            f.write('\n')
            # break

def process_data(tmp_location):
    with open(tmp_location+'/current_kg.json', 'w') as f:
        current_kg = json.load(f)

    current_kg['postEntityToCSKTripleIndex'] = {}
    current_kg['postEntityToOtherCSKTripleEntities'] = {}
    index = 0
    for triple in current_kg['csk_triples']:
        firstEntity = triple.split('$$')[0]
        secondEntity = triple.split('$$')[2].strip()
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

    convert_data('dialog_dataset/formatted_train.json', 'data/none/trainset.txt', current_kg)
    convert_data('dialog_dataset/formatted_dev.json', 'data/none/validset.txt', current_kg)
    convert_data('dialog_dataset/formatted_test.json', 'data/none/testset.txt', current_kg)


prepare_kg('kgs/conceptnet.txt', 'data/conceptnet')
# process_data('data/conceptnet')

