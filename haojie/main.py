import ujson as json


import argparse
import os
import time
import torch
import ujson as json
from dialogue.Trainer import Trainer
from dialogue.io.DialogueDataset import DialogueDatasetIterator
from dialogue.toolbox.config import Config
from dialogue.toolbox.logging import init_logger
from dialogue.toolbox.optim import Optim
from dialogue.toolbox.utils import get_num_parameters
from dialogue.toolbox.vocab import PAD_WORD, BOS_WORD, EOS_WORD
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def train_model(train_opt):
    total_st = time.time()
    meta_opt = train_opt.meta
    optim_opt = train_opt.optimizer
    model_opt = train_opt.model
    if not os.path.exists(meta_opt.data_cache_dir):
        os.makedirs(meta_opt.data_cache_dir)
    if not os.path.exists(meta_opt.save_log):
        os.makedirs(meta_opt.save_log)
    if not os.path.exists(meta_opt.save_model):
        os.makedirs(meta_opt.save_model)
    if not os.path.exists(meta_opt.save_results):
        os.makedirs(meta_opt.save_results)
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger = init_logger(meta_opt.save_log + "%s.log" % current_time)
    logger.info("Initializing...")
    if meta_opt.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(meta_opt.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    vocabs = torch.load(meta_opt.vocab_path)

    model_opt.word_vocab_size = len(vocabs["word"])
    model_opt.aser_vocab_size = len(vocabs["aser"])
    model_opt.aser_event_vocab_size = len(vocabs["aser_event"])
    model_opt.aser_relation_vocab_size = len(vocabs["aser_relation"])

    model_opt.omcs_vocab_size = len(vocabs["omcs"])
    model_opt.omcs_event_vocab_size = len(vocabs["omcs_event"])
    model_opt.omcs_relation_vocab_size = len(vocabs["omcs_relation"])

    model_opt.knowly_vocab_size = len(vocabs["knowlywood"])
    model_opt.knowly_event_vocab_size = len(vocabs["knowlywood_event"])
    model_opt.knowly_relation_vocab_size = len(vocabs["knowlywood_relation"])

    meta_opt.pad_idx = vocabs["word"].to_idx(PAD_WORD)
    meta_opt.bos_idx = vocabs["word"].to_idx(BOS_WORD)
    meta_opt.eos_idx = vocabs["word"].to_idx(EOS_WORD)

    optimizer = Optim(
        optim_opt.optim, optim_opt.learning_rate, optim_opt.max_grad_norm,
        lr_decay=optim_opt.learning_rate_decay,
        start_decay_steps=optim_opt.start_decay_steps,
        decay_steps=optim_opt.decay_steps,
        beta1=optim_opt.adam_beta1,
        beta2=optim_opt.adam_beta2,
        adam_eps=optim_opt.adam_eps,
        adagrad_accum=optim_opt.adagrad_accumulator_init,
        decay_method=optim_opt.decay_method,
        warmup_steps=optim_opt.warmup_steps,
        model_size=model_opt.rnn_hidden_size)

    train_iter = DialogueDatasetIterator(
        file_path=os.path.join(meta_opt.data_dir, "train.json"),
        file_cache_path=os.path.join(meta_opt.data_cache_dir, "train.pt"),
        vocabs=vocabs, epochs=meta_opt.epochs, batch_size=meta_opt.batch_size,
        is_train=True, n_workers=meta_opt.n_workers, use_cuda=meta_opt.use_cuda, opt=meta_opt)
    valid_iter = DialogueDatasetIterator(
        file_path=os.path.join(meta_opt.data_dir, "valid.json"),
        file_cache_path=os.path.join(meta_opt.data_cache_dir, "valid.pt"),
        vocabs=vocabs, epochs=meta_opt.epochs, batch_size=meta_opt.valid_batch_size,
        is_train=False, n_workers=meta_opt.n_workers, use_cuda=meta_opt.use_cuda, opt=meta_opt)

    trainer = Trainer(train_iter, valid_iter, vocabs, optimizer, train_opt, logger)
    logger.info(trainer.model)
    logger.info("Word vocab size: %d" % len(vocabs["word"]))
    logger.info("Aser vocab size: %d" % len(vocabs["aser"]))
    logger.info("Omcs vocab size: %d" % len(vocabs["omcs"]))
    logger.info("KnowlyWood vocab size: %d" % len(vocabs["knowlywood"]))
    logger.info("Total parameters: %d " % get_num_parameters(trainer.model))
    logger.info("Trainable parameters: %d " % get_num_parameters(trainer.model, trainable=True))
    logger.info("Start training...")
    trainer.train()
    logger.info("Total training time: %.2f s" % (time.time() - total_st))


def model_evaluate(model_path, inp_path):
    model_file = torch.load(model_path)
    train_opt = model_file["train_opt"]
    vocabs = model_file["vocabs"]
    test_data = list()
    with open(inp_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    meta_opt = train_opt.meta
    test_iter = DialogueDatasetIterator(
        file_path=inp_path, vocabs=vocabs,
        epochs=meta_opt.epochs, batch_size=1,
        is_train=False, n_workers=meta_opt.n_workers,
        use_cuda=meta_opt.use_cuda, opt=meta_opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(meta_opt.gpu)

    model = model_file["model"]
    model.eval()
    model.flatten_parameters()

    pred_list = []
    total_loss = 0
    total_word_num = 0
    for batch in tqdm(test_iter):
        result_dict = model.run_batch(batch)
        total_loss += result_dict["loss"].item()
        total_word_num += result_dict["num_words"]
        preds_idx = model.predict_batch(
            batch, max_len=20, beam_size=5, eos_val=vocabs["word"].to_idx(EOS_WORD))
        preds = [[vocabs["word"].to_word(t) for t in item] for item in preds_idx]
        pred_list.extend(preds)

    per_word_loss = total_loss / total_word_num
    s1 = "Valid, Loss: {:.2f}, PPL: {:.2f}".format(np.log(model_file["score"]), model_file["score"])
    s2 = "Test, Loss: {:.2f}, PPL: {:.2f}".format(per_word_loss, np.exp(per_word_loss))
    print(s1)
    print(s2)
    overall_bleu_1_score = 0
    overall_bleu_2_score = 0
    overall_bleu_3_score = 0
    overall_bleu_4_score = 0
    for i, raw_predict in enumerate(pred_list):
        tmp_predict = list()
        for w in raw_predict:
            if w == BOS_WORD or w == EOS_WORD:
                continue
            tmp_predict.append(w)
        overall_bleu_1_score += sentence_bleu([test_data[i]['response'].split(' ')], tmp_predict, weights=(1, 0, 0, 0))
        overall_bleu_2_score += sentence_bleu([test_data[i]['response'].split(' ')], tmp_predict, weights=(0.5, 0.5, 0, 0))
        overall_bleu_3_score += sentence_bleu([test_data[i]['response'].split(' ')], tmp_predict, weights=(0.33, 0.33, 0.34, 0))
        overall_bleu_4_score += sentence_bleu([test_data[i]['response'].split(' ')], tmp_predict, weights=(0.25, 0.25, 0.25, 0.25))
    print('Average bleu score:', 'bleu1:', overall_bleu_1_score / len(pred_list), 'bleu2:',
          overall_bleu_2_score / len(pred_list), 'bleu3:', overall_bleu_3_score / len(pred_list), 'bleu4:',
          overall_bleu_4_score / len(pred_list))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_omcs2seq.json",
                        help="configuration file path")
    parser.add_argument("--folder", type=str, default="data/conceptnet",
                        help="which data to train on")
    parser.add_argument("--gpu", type=str, default="0",
                        help="which data to use")
    args = parser.parse_args()
    raw_config = json.load(open(args.config))
    raw_config['meta']['data_dir'] = args.folder
    raw_config['meta']['vocab_path'] = args.folder + '/vocab.pt'
    raw_config['meta']['data_cache_dir'] = args.folder + '/cache/data/'
    raw_config['meta']['save_model'] = args.folder + '/cache/model'
    raw_config['meta']['save_results'] = args.folder + '/cache/results'
    raw_config['meta']['save_log'] = args.folder + '/cache/log'
    raw_config['meta']['gpu'] = args.gpu
    opt = Config(raw_config)
    train_model(opt)

    print('start to evaluate')
    model_evaluate(args.folder + '/cache/model/best_model.pt', args.folder + '/test.json')

print('end')
