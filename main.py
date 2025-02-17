import numpy as np
import tensorflow as tf
import json
from tensorflow.python.framework import constant_op
import sys
import math
import os
import time
import random
from nltk.translate.bleu_score import sentence_bleu

random.seed(time.time())
from model import Model, _START_VOCAB
import argparse
from tqdm import tqdm

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")
tf.app.flags.DEFINE_integer("num_entities", 21471, "entitiy vocabulary size.")
tf.app.flags.DEFINE_integer("num_relations", 21, "relation size.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("trans_units", 50, "Size of trans embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 4, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data/none", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", False, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "test", "Set filename of inference")
tf.app.flags.DEFINE_string("gpu", "0", "which gpu to use")

FLAGS = tf.app.flags.FLAGS
if FLAGS.train_dir[-1] == '/': FLAGS.train_dir = FLAGS.train_dir[:-1]
csk_triples, csk_entities, kb_dict = [], [], []


def prepare_data(path, is_train=True):
    global csk_entities, csk_triples, kb_dict

    with open(path + '/current_kg.json', 'r') as f:
        d = json.load(f)

    csk_triples = d['csk_triples']
    csk_entities = d['csk_entities']
    raw_vocab = d['vocab_dict']
    kb_dict = d['dict_csk']

    data_train, data_dev, data_test = [], [], []

    if is_train:
        with open('%s/trainset.txt' % path) as f:
            for idx, line in enumerate(f):
                # if idx == 100000: break
                if idx % 100000 == 0: print('read train file line %d' % idx)
                data_train.append(json.loads(line))

    with open('%s/validset.txt' % path) as f:
        for line in f:
            data_dev.append(json.loads(line))

    with open('%s/testset.txt' % path) as f:
        for line in f:
            data_test.append(json.loads(line))

    return raw_vocab, data_train, data_dev, data_test


def build_vocab(path, raw_vocab, trans='transE'):
    print("Creating word vocabulary...")
    vocab_list = _START_VOCAB + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]

    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    try:
        with open('%s/entity.txt' % path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                e = line.strip()
                entity_list.append(e)
    except FileNotFoundError:
        print('we do not find entity.txt')
        pass

    print("Creating relation vocabulary...")
    relation_list = []
    try:
        with open('%s/relation.txt' % path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                r = line.strip()
                relation_list.append(r)
    except FileNotFoundError:
        print('we do not find relation.txt')
        pass

    print("Loading word vectors...")
    vectors = {}
    try:
        with open('glove.840B.300d.txt', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print("    processing line %d" % i)
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ') + 1:]
                vectors[word] = vector
    except FileNotFoundError:
        print('we do not find glove.840B.300d.txt')
        pass

    embed = []
    for word in vocab_list:
        if word in vectors:
            # print(len(vectors[word].split()))
            tmp_vector = list()
            for w in vectors[word].split():
                tmp_vector.append(float(w))
            vector = np.array(tmp_vector, dtype=np.float32)
        else:
            # print('We cannot find word:', word)
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)

    print("Loading entity vectors...")
    entity_embed = []
    try:
        # with open('%s/entity_%s.txt' % (path, trans)) as f:
        #     for i, line in enumerate(f):
        #         s = line.strip().split('\t')
        #         entity_embed.append(map(float, s))
        with open(path + '/entity_vector.json', 'r') as f:
            entity_embed = json.load(f)
    except FileNotFoundError:
        print('we do not find entity vector')
        pass

    print("Loading relation vectors...")
    relation_embed = []
    try:
        # with open('%s/relation_%s.txt' % (path, trans)) as f:
        #     for i, line in enumerate(f):
        #         s = line.strip().split('\t')
        #         relation_embed.append(s)
        with open(path + '/relation_vector.json', 'r') as f:
            relation_embed = json.load(f)
    except FileNotFoundError:
        print('we do not find the relation vector')
        pass

    entity_relation_embed = np.array(entity_embed + relation_embed, dtype=np.float32)
    entity_embed = np.array(entity_embed, dtype=np.float32)
    relation_embed = np.array(relation_embed, dtype=np.float32)

    return vocab_list, embed, entity_list, entity_embed, relation_list, relation_embed, entity_relation_embed


def gen_batched_data(data):
    global csk_entities, csk_triples, kb_dict
    encoder_len = max([len(item['post']) for item in data]) + 1
    decoder_len = max([len(item['response']) for item in data]) + 1
    triple_num = max([len(item['all_triples']) for item in data]) + 1
    try:
        triple_len = max([len(tri) for item in data for tri in item['all_triples']])
    except:
        triple_len = 1
    max_length = 20
    posts, responses, posts_length, responses_length = [], [], [], []
    entities, triples, matches, post_triples, response_triples = [], [], [], [], []
    match_entities, all_entities = [], []
    match_triples, all_triples = [], []
    NAF = ['_NAF_H', '_NAF_R', '_NAF_T']

    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)

    def padding_triple(triple, num, l):
        newtriple = []
        triple = [[NAF]] + triple
        for tri in triple:
            newtriple.append(tri + [['_PAD_H', '_PAD_R', '_PAD_T']] * (l - len(tri)))
        pad_triple = [['_PAD_H', '_PAD_R', '_PAD_T']] * l
        return newtriple + [pad_triple] * (num - len(newtriple))

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post']) + 1)
        responses_length.append(len(item['response']) + 1)
        all_triples.append(
            padding_triple([[csk_triples[x].split('$$') for x in triple] for triple in item['all_triples']], triple_num,
                           triple_len))
        post_triples.append([[x] for x in item['post_triples']] + [[0]] * (encoder_len - len(item['post_triples'])))
        response_triples.append(
            [NAF] + [NAF if x == -1 else csk_triples[x].split('$$') for x in item['response_triples']] + [NAF] * (
                    decoder_len - 1 - len(item['response_triples'])))
        match_index = []
        for idx, x in enumerate(item['match_index']):
            _index = [-1] * triple_num
            if x[0] == -1 and x[1] == -1:
                match_index.append(_index)
            else:
                _index[x[0]] = x[1]
                t = all_triples[-1][x[0]][x[1]]
                assert (t == response_triples[-1][idx + 1])
                match_index.append(_index)
        match_triples.append(match_index + [[-1] * triple_num] * (decoder_len - len(match_index)))

        if not FLAGS.is_train:
            entity = [['_NONE'] * triple_len]
            for ent in item['all_entities']:
                entity.append([csk_entities[int(x)] for x in ent] + ['_NONE'] * (triple_len - len(ent)))
            entities.append(entity + [['_NONE'] * triple_len] * (triple_num - len(entity)))
        else:
            entity = [['_NONE'] * triple_len]
            for ent in item['all_entities']:
                entity.append([csk_entities[int(x)] for x in ent] + ['_NONE'] * (triple_len - len(ent)))
            entities.append(entity + [['_NONE'] * triple_len] * (triple_num - len(entity)))

    batched_data = {'posts': np.array(posts),
                    'responses': np.array(responses),
                    'posts_length': posts_length,
                    'responses_length': responses_length,
                    'triples': np.array(all_triples),
                    'entities': np.array(entities),
                    'posts_triple': np.array(post_triples),
                    'responses_triple': np.array(response_triples),
                    'match_triples': np.array(match_triples)}

    return batched_data


def train(model, sess, data_train):
    batched_data = gen_batched_data(data_train)
    outputs = model.step_decoder(sess, batched_data)
    return np.sum(outputs[0])


def generate_summary(model, sess, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(selected_data)
    summary = model.step_decoder(sess, batched_data, forward_only=True, summary=True)[-1]
    return summary


def evaluate(model, sess, data_dev, summary_writer):
    print('number of evaluation example:', len(data_dev))
    loss = np.zeros((1,))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True)
        loss += np.sum(outputs[0])
        st, ed = ed, ed + FLAGS.batch_size
        times += 1
    loss /= len(data_dev)
    summary = tf.Summary()
    summary.value.add(tag='decoder_loss/dev', simple_value=loss)
    summary.value.add(tag='perplexity/dev', simple_value=np.exp(loss))
    summary_writer.add_summary(summary, model.global_step.eval())
    print('    perplexity on dev set: %.2f' % np.exp(loss))


def get_steps(train_dir):
    a = os.walk(train_dir)
    for root, dirs, files in a:
        if root == train_dir:
            filenames = files

    steps, metafiles, datafiles, indexfiles = [], [], [], []
    for filename in filenames:
        if 'meta' in filename:
            metafiles.append(filename)
        if 'data' in filename:
            datafiles.append(filename)
        if 'index' in filename:
            indexfiles.append(filename)

    metafiles.sort()
    datafiles.sort()
    indexfiles.sort(reverse=True)

    for f in indexfiles:
        steps.append(int(f[11:-6]))

    return steps


def new_test(sess, saver, data_dev, setnum=5000):
    results = []
    loss = []
    evaluation_data_by_batch = list()
    for j in range(int(train_len / FLAGS.batch_size) + 1):
        if len(data_dev[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]) > 0:
            evaluation_data_by_batch.append(data_dev[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size])
    print('start to generate response')
    for tmp_data in tqdm(evaluation_data_by_batch):
        # selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(tmp_data)
        # print(batched_data)
        responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'],
                                       {'enc_inps:0': batched_data['posts'],
                                        'enc_lens:0': batched_data['posts_length'],
                                        'dec_inps:0': batched_data['responses'],
                                        'dec_lens:0': batched_data['responses_length'],
                                        'entities:0': batched_data['entities'],
                                        'triples:0': batched_data['triples'],
                                        'match_triples:0': batched_data['match_triples'],
                                        'enc_triples:0': batched_data['posts_triple'],
                                        'dec_triples:0': batched_data['responses_triple']})
        loss += [x for x in ppx_loss]

        for response in responses:
            result = []
            for token in response:
                if token != '_EOS':
                    result.append(token.decode("utf-8"))
                else:
                    break
            results.append(result)

    overall_bleu_1_score = 0
    overall_bleu_2_score = 0
    overall_bleu_3_score = 0
    overall_bleu_4_score = 0
    overall_bleu_score = 0

    # print('start to calculate the bleu score')
    for i, tmp_response in enumerate(results):
        gold_answer = data_dev[i]['response']
        print(gold_answer)
        print(tmp_response)
        # tmp_response = tmp_response + gold_answer
        overall_bleu_1_score += sentence_bleu([gold_answer], tmp_response, weights=(1, 0, 0, 0))
        overall_bleu_2_score += sentence_bleu([gold_answer], tmp_response, weights=(0, 1, 0, 0))
        overall_bleu_3_score += sentence_bleu([gold_answer], tmp_response, weights=(0, 0, 1, 0))
        overall_bleu_4_score += sentence_bleu([gold_answer], tmp_response, weights=(0, 0, 0, 1))
        overall_bleu_score += sentence_bleu([gold_answer], tmp_response, weights=(0.25, 0.25, 0.25, 0.25))
        # overall_bleu_score += tmp_bleu_score
    print('Average bleu score:', 'bleu1:', overall_bleu_1_score / len(results), 'bleu2:',
          overall_bleu_2_score / len(results), 'bleu3:', overall_bleu_3_score / len(results), 'bleu4:',
          overall_bleu_4_score / len(results), 'average:', overall_bleu_score/len(results))

    # match_entity_sum = [m / setnum for m in match_entity_sum] + [sum(match_entity_sum) / len(data_dev)]
    losses = [np.sum(loss[x:x + setnum]) / float(setnum) for x in range(0, setnum * 4, setnum)] + [
        np.sum(loss) / float(setnum * 4)]
    losses = [np.exp(x) for x in losses]

    # def show(x):
    #     return ', '.join([str(v) for v in x])

    print('perplexity:', sum(losses) / len(losses))

    # print('perplexity: %s\n\tmatch_entity_rate: %s\n%s\n\n' % (show(losses), show(match_entity_sum), '=' * 50))
    # print(
    #     'perplexity: %s\n\tmatch_entity_rate: %s\n\n' % (show(losses), show(match_entity_sum)))


def test(sess, saver, data_dev, setnum=5000):
    # with open('%s/stopwords' % 'data' + FLAGS.data_dir) as f:
    #     stopwords = json.loads(f.readline())
    stopwords = list()
    steps = get_steps(FLAGS.train_dir)
    low_step = 00000
    high_step = 800000
    for step in [step for step in steps if step > low_step and step < high_step]:
        # model_path = '%s/checkpoint.tmp' % FLAGS.train_dir
        model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        print('restore from %s' % model_path)
        try:
            saver.restore(sess, model_path)
        except:
            continue
        st, ed = 0, FLAGS.batch_size
        results = []
        loss = []

        evaluation_data_by_batch = list()
        for j in range(int(train_len / FLAGS.batch_size) + 1):
            if len(data_dev[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]) > 0:
                evaluation_data_by_batch.append(data_dev[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size])

        for tmp_data in tqdm(evaluation_data_by_batch):
            batched_data = gen_batched_data(tmp_data)
            print(batched_data)
            responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'],
                                           {'enc_inps:0': batched_data['posts'],
                                            'enc_lens:0': batched_data['posts_length'],
                                            'dec_inps:0': batched_data['responses'],
                                            'dec_lens:0': batched_data['responses_length'],
                                            'entities:0': batched_data['entities'],
                                            'triples:0': batched_data['triples'],
                                            'match_triples:0': batched_data['match_triples'],
                                            'enc_triples:0': batched_data['posts_triple'],
                                            'dec_triples:0': batched_data['responses_triple']})
            loss += [x for x in ppx_loss]

            for response in responses:
                result = []
                for token in response:
                    if token != '_EOS':
                        result.append(token)
                    else:
                        break
                results.append(result)
        match_entity_sum = [.0] * 4
        cnt = 0
        for post, response, result, match_triples, triples, entities in zip([data['post'] for data in data_dev],
                                                                            [data['response'] for data in data_dev],
                                                                            results, [data['match_triples'] for data in
                                                                                      data_dev],
                                                                            [data['all_triples'] for data in data_dev],
                                                                            [data['all_entities'] for data in
                                                                             data_dev]):
            setidx = cnt / setnum
            result_matched_entities = []
            triples = [csk_triples[tri] for triple in triples for tri in triple]
            match_triples = [csk_triples[triple] for triple in match_triples]
            entities = [csk_entities[x] for entity in entities for x in entity]
            matches = [x for triple in match_triples for x in [triple.split('$$')[0], triple.split('$$')[2]] if
                       x in response]

            for word in result:
                if word not in stopwords and word in entities:
                    result_matched_entities.append(word)
            match_entity_sum[setidx] += len(set(result_matched_entities))
            cnt += 1

        overall_bleu_score = 0
        for i, tmp_response in enumerate(results):
            gold_answer = data_dev[i]['response']
            tmp_bleu_score = sentence_bleu([gold_answer], tmp_response)
            overall_bleu_score += tmp_bleu_score
        print('Average bleu score')

        match_entity_sum = [m / setnum for m in match_entity_sum] + [sum(match_entity_sum) / len(data_dev)]
        losses = [np.sum(loss[x:x + setnum]) / float(setnum) for x in range(0, setnum * 4, setnum)] + [
            np.sum(loss) / float(setnum * 4)]
        losses = [np.exp(x) for x in losses]

        def show(x):
            return ', '.join([str(v) for v in x])

        print('model: %d\n\tperplexity: %s\n\tmatch_entity_rate: %s\n%s\n\n' % (
            step, show(losses), show(match_entity_sum), '=' * 50))
        print(
            'model: %d\n\tperplexity: %s\n\tmatch_entity_rate: %s\n\n' % (step, show(losses), show(match_entity_sum)))

    # return results


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
# parser.add_argument('--data_dir', type=str, default='none', help='which dataset to train')
# args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        raw_vocab, data_train, data_dev, data_test = prepare_data('data/' + FLAGS.data_dir)
        vocab, embed, entity_vocab, entity_embed, relation_vocab, relation_embed, entity_relation_embed = build_vocab(
            'data/' + FLAGS.data_dir, raw_vocab)
        FLAGS.num_entities = len(entity_vocab)
        print(FLAGS.__flags)
        model = Model(
            FLAGS.symbols,
            FLAGS.embed_units,
            FLAGS.units,
            FLAGS.layers,
            embed,
            entity_relation_embed,
            num_entities=len(entity_vocab) + len(relation_vocab),
            num_trans_units=FLAGS.trans_units)
        if tf.train.get_checkpoint_state(FLAGS.data_dir):
            print("Reading model parameters from %s" % FLAGS.data_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.data_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                                              constant_op.constant(list(range(len(vocab))), dtype=tf.int64))
            sess.run(op_in)
            op_out = model.index2symbol.insert(constant_op.constant(
                list(range(len(vocab))), dtype=tf.int64), constant_op.constant(vocab))
            sess.run(op_out)
            op_in = model.entity2index.insert(constant_op.constant(entity_vocab + relation_vocab),
                                              constant_op.constant(list(range(len(entity_vocab) + len(relation_vocab))),
                                                                   dtype=tf.int64))
            sess.run(op_in)
            op_out = model.index2entity.insert(constant_op.constant(
                list(range(len(entity_vocab) + len(relation_vocab))), dtype=tf.int64),
                constant_op.constant(entity_vocab + relation_vocab))
            sess.run(op_out)
        # print("Created model with fresh parameters.")
        # tf.global_variables_initializer().run()
        # op_in = model.symbol2index.insert(constant_op.constant(vocab),
        #                                   constant_op.constant(list(range(len(vocab))), dtype=tf.int64))
        # sess.run(op_in)
        # op_out = model.index2symbol.insert(constant_op.constant(
        #     list(range(len(vocab))), dtype=tf.int64), constant_op.constant(vocab))
        # sess.run(op_out)
        # op_in = model.entity2index.insert(constant_op.constant(entity_vocab + relation_vocab),
        #                                   constant_op.constant(list(range(len(entity_vocab) + len(relation_vocab))),
        #                                                        dtype=tf.int64))
        # sess.run(op_in)
        # op_out = model.index2entity.insert(constant_op.constant(
        #     list(range(len(entity_vocab) + len(relation_vocab))), dtype=tf.int64),
        #     constant_op.constant(entity_vocab + relation_vocab))
        # sess.run(op_out)

        if FLAGS.log_parameters:
            model.print_parameters()

        summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
        loss_step, time_step = np.zeros((1,)), .0
        previous_losses = [1e18] * 3
        # data_train = data_train[:50]
        # data_train = data_test
        data_train = data_train[:100]
        train_len = len(data_train)
        number_of_iteration = 50

        # print('Dev set:')
        # new_test(sess, model.saver, data_dev)
        # evaluate(model, sess, data_dev, summary_writer)
        # print('Test set:')
        # new_test(sess, model.saver, data_test)
        for i in range(number_of_iteration):
            print('Current data resource:', FLAGS.data_dir)
            print('current iteration:', i + 1, '/', number_of_iteration)
            st, ed = 0, FLAGS.batch_size * FLAGS.per_checkpoint
            random.shuffle(data_train)
            train_data_by_batch = list()
            for j in range(int(train_len / FLAGS.batch_size) + 1):
                if len(data_train[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]) > 0:
                    train_data_by_batch.append(data_train[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size])
            for tmp_train_data in tqdm(train_data_by_batch):
                # print('number of example:', len(tmp_train_data))
                loss_step += train(model, sess, tmp_train_data)
            loss_step /= len(train_data_by_batch)
            show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
            print("global step %d learning rate %.4f loss %f perplexity %s"
                  % (model.global_step.eval(), model.lr, loss_step, show(np.exp(loss_step))))
            model.saver.save(sess, '%s/checkpoint.tmp' % FLAGS.data_dir,
                             global_step=model.global_step)
            # print('Dev set:')
            # new_test(sess, model.saver, data_dev)
            # evaluate(model, sess, data_dev, summary_writer)
            print('Test set:')
            new_test(sess, model.saver, data_train)
            # evaluate(model, sess, data_test, summary_writer)
            # model.saver_epoch.save(sess, '%s/epoch/checkpoint' % FLAGS.train_dir, global_step=model.global_step)

            # while st < train_len:
            #     start_time = time.time()
            #
            #     for batch in range(st, ed, FLAGS.batch_size):
            #         loss_step += train(model, sess, data_train[batch:batch+FLAGS.batch_size]) / (ed - st)
            #
            #
            #
            #     model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir,
            #             global_step=model.global_step)
            #     summary = tf.Summary()
            #     summary.value.add(tag='decoder_loss/train', simple_value=loss_step)
            #     summary.value.add(tag='perplexity/train', simple_value=np.exp(loss_step))
            #     summary_writer.add_summary(summary, model.global_step.eval())
            #     summary_model = generate_summary(model, sess, data_train)
            #     summary_writer.add_summary(summary_model, model.global_step.eval())
            #     evaluate(model, sess, data_dev, summary_writer)
            #     previous_losses = previous_losses[1:]+[np.sum(loss_step)]
            #     loss_step, time_step = np.zeros((1, )), .0
            #     st, ed = ed, min(train_len, ed + FLAGS.batch_size * FLAGS.per_checkpoint)
        # if FLAGS.inference_version == 0:
        #     model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        # else:
        #     model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        # print('restore from %s' % model_path)
        # model.saver.restore(sess, model_path)
        # saver = model.saver
        # test(sess, model.saver, data_test, setnum=5000)
    # else:
    #     model = Model(
    #             FLAGS.symbols,
    #             FLAGS.embed_units,
    #             FLAGS.units,
    #             FLAGS.layers,
    #             embed=None,
    #             num_entities=FLAGS.num_entities+FLAGS.num_relations,
    #             num_trans_units=FLAGS.trans_units)
    #
    #     if FLAGS.inference_version == 0:
    #         model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    #     else:
    #         model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
    #     print('restore from %s' % model_path)
    #     model.saver.restore(sess, model_path)
    #     saver = model.saver
    #
    #     raw_vocab, data_train, data_dev, data_test = prepare_data('data/' + FLAGS.data_dir, is_train=False)
    #
    #     test(sess, saver, data_test, setnum=5000)
