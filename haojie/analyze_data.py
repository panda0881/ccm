import ujson as json
from tqdm import tqdm
from dialogue.toolbox.vocab import Vocabulary, UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, get_pretrained_embedding
from collections import Counter
import torch
import argparse

WORD_VOCAB_SIZE = 15000
ASER_VOCAB_SIZE = 40000
ASER_EVENT_VOCAB_SIZE = 30000
OMCS_VOCAB_SIZE = 30000
OMCS_EVENT_VOCAB_SIZE = 30000
KNOWLY_VOCAB_SIZE = 45000
KNOWLY_EVENT_VOCAB_SIZE = 40000


def build_vocabs(counters):
    word_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])
    word_vocab.build_from_counter(counters["word"], WORD_VOCAB_SIZE)

    omcs_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    omcs_vocab.build_from_counter(counters["omcs"], OMCS_VOCAB_SIZE)

    omcs_event_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    omcs_event_vocab.build_from_counter(counters["omcs_event"], OMCS_EVENT_VOCAB_SIZE)

    omcs_rel_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    omcs_rel_vocab.build_from_counter(counters["omcs_relation"], 21)

    vocabs = {
        "word": word_vocab,
        "pre_word_emb": None,
        "omcs": omcs_vocab,
        "omcs_event": omcs_event_vocab,
        "omcs_relation": omcs_rel_vocab
    }
    return vocabs


def analyze_data(kg_path, output_path):
    print('We are working on:', kg_path)
    tmp_knowledge = list()

    # if kg_path != '/home/guest/hzhangal/ccm/kgs/conceptnet.txt':
    #     with open('/home/guest/hzhangal/ccm/kgs/conceptnet.txt', 'r', encoding='utf-8') as f:
    #         for line in f:
    #             tmp_words = line[:-1].split('\t')
    #             tmp_head = tmp_words[1].replace('$', '')
    #             tmp_relation = tmp_words[0].replace('$', '')
    #             tmp_tail = tmp_words[2].replace('$', '')
    #             tmp_knowledge.append(tmp_head+'$'+tmp_relation+'$'+tmp_tail)

    with open(kg_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_words = line[:-1].split('\t')
            tmp_head = tmp_words[1].replace('$', '')
            tmp_relation = tmp_words[0].replace('$', '')
            tmp_tail = tmp_words[2].replace('$', '')
            tmp_knowledge.append(tmp_head + '$' + tmp_relation + '$' + tmp_tail)

    tmp_knowledge = list(set(tmp_knowledge))

    all_knowledge = list()
    all_knowledge_dict = dict()
    for tmp_k in tmp_knowledge:
        all_knowledge.append({'head': tmp_k.split('$')[0], 'tail': tmp_k.split('$')[2], 'relation': tmp_k.split('$')[1]})
        if tmp_k.split('$')[0] not in all_knowledge_dict:
            all_knowledge_dict[tmp_k.split('$')[0]] = list()
        all_knowledge_dict[tmp_k.split('$')[0]].append(tmp_k)
        if tmp_k.split('$')[2] not in all_knowledge_dict:
            all_knowledge_dict[tmp_k.split('$')[2]] = list()
        all_knowledge_dict[tmp_k.split('$')[2]].append(tmp_k)


    # with open('/home/guest/hzhangal/ccm/dialog_dataset/formatted_train.json', 'r', encoding='utf-8') as f:
    #     train_data = json.load(f)
    #
    # with open('/home/guest/hzhangal/ccm/dialog_dataset/formatted_dev.json', 'r', encoding='utf-8') as f:
    #     dev_data = json.load(f)

    with open('/home/guest/hzhangal/ccm/dialog_dataset/formatted_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # print('We are working on training data')
    # new_train_data = list()
    # for tmp_example in tqdm(train_data):
    #     new_example = dict()
    #     new_example['post'] = tmp_example['post']
    #     new_example['response'] = tmp_example['response']
    #     new_example['omcs_triples'] = list()
    #     # for tmp_k in all_knowledge:
    #     #     if tmp_k['head'] in tmp_example['post'] or tmp_k['tail'] in tmp_example['post']:
    #     #         new_example['omcs_triplets'].append(tmp_k['head']+'$'+tmp_k['relation']+'$'+tmp_k['tail'])
    #     for tmp_e in all_knowledge_dict:
    #         if tmp_e in tmp_example['post']:
    #             new_example['omcs_triples'] += all_knowledge_dict[tmp_e]
    #     new_example['omcs_triples'] = list(set(new_example['omcs_triples']))
    #     new_train_data.append(new_example)
    #
    # print('We are working on dev data')
    # new_dev_data = list()
    # for tmp_example in tqdm(dev_data):
    #     new_example = dict()
    #     new_example['post'] = tmp_example['post']
    #     new_example['response'] = tmp_example['response']
    #     new_example['omcs_triples'] = list()
    #     # for tmp_k in all_knowledge:
    #     #     if tmp_k['head'] in tmp_example['post'] or tmp_k['tail'] in tmp_example['post']:
    #     #         new_example['omcs_triplets'].append(tmp_k['head']+'$'+tmp_k['relation']+'$'+tmp_k['tail'])
    #     for tmp_e in all_knowledge_dict:
    #         if tmp_e in tmp_example['post']:
    #             new_example['omcs_triples'] += all_knowledge_dict[tmp_e]
    #     new_example['omcs_triples'] = list(set(new_example['omcs_triples']))
    #     new_dev_data.append(new_example)

    print('We are working on test data')
    all_count = 0
    match_count = 0
    new_test_data = list()
    for tmp_example in tqdm(test_data):
        for tmp_k in all_knowledge:
            if tmp_k['head'] in tmp_example['post']:
                all_count += 1
                if tmp_k['tail'] in tmp_example['response']:
                    match_count += 1
            if tmp_k['tail'] in tmp_example['post']:
                all_count += 1
                if tmp_k['head'] in tmp_example['response']:
                    match_count += 1
    print('all match:', all_count, '/', len(test_data), all_count/len(test_data))
    print('correct match:', match_count, '/', len(test_data), match_count/len(test_data))






parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='all',
                        help="choose which gpu to use")
parser.add_argument("--output", type=str, default='/home/guest/hzhangal/ccm/haojie/data/auto_conceptnet_1_percent/',
                        help="choose which gpu to use")
args = parser.parse_args()


if args.input == 'all':
    # analyze_data('/home/guest/hzhangal/ccm/kgs/conceptnet.txt', '/home/guest/hzhangal/ccm/haojie/data/conceptnet/')
    analyze_data('/home/guest/hzhangal/ccm/kgs/COMET_original_1.txt', '/home/guest/hzhangal/ccm/haojie/data/COMET_original_1/')
    analyze_data('/home/guest/hzhangal/ccm/kgs/COMET_external_10.txt', '/home/guest/hzhangal/ccm/haojie/data/COMET_external_10/')
    analyze_data('/home/guest/hzhangal/ccm/kgs/LAMA_original_1.txt', '/home/guest/hzhangal/ccm/haojie/data/LAMA_original_1/')
    analyze_data('/home/guest/hzhangal/ccm/kgs/LAMA_external_10.txt', '/home/guest/hzhangal/ccm/haojie/data/LAMA_external_10/')
    analyze_data('/home/guest/hzhangal/ccm/kgs/auto_conceptnet_1_percent.txt', '/home/guest/hzhangal/ccm/haojie/data/auto_conceptnet_1_percent/')
else:
    analyze_data(args.input, args.output)

# with open('/home/guest/hzhangal/ccm/dialog_dataset/formatted_test.json', 'r', encoding='utf-8') as f:
#     test_data = json.load(f)
#
# with open('test.response.txt', 'w', encoding='utf-8') as f:
#     for tmp_example in test_data:
#         f.write(tmp_example['response'])
#         f.write('\n')


print('end')
