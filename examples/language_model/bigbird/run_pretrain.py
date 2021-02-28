# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import itertools
import logging
import os
import random
import time
import h5py
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BigBirdForPretraining, BigBirdModel, BigBirdPretrainingCriterion
from paddlenlp.transformers import BigBirdTokenizer, LinearDecayWithWarmup, create_bigbird_rand_mask_idx_list
from paddlenlp.utils.log import logger
import time

MODEL_CLASSES = {"bigbird": (BigBirdForPretraining, BigBirdTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="bigbird",
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="bigbird-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    # parser.add_argument(
    #     "--input_dir",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="The input directory where the data will be read from.", )
    # parser.add_argument(
    #     "--output_dir",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )

    parser.add_argument(
        "--max_predictions_per_seq",
        default=80,
        type=int,
        help="The maximum total of masked tokens in input sequence")

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epoches for training.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='~/',
        help="vocab file used to tokenize text")
    parser.add_argument(
        "--vocab_model_file",
        type=str,
        default='sentencepiece_gpt2.model',
        help="vocab model file used to tokenize text")
    parser.add_argument(
        "--max_encoder_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after SentencePiece tokenization."
    )
    parser.add_argument(
        "--num_train_steps",
        default=10000,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="")
    parser.add_argument("--hidden_size", type=int, default=768, help="")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=3072)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--normalize_before", type=bool, default=False)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--num_rand_blocks", type=int, default=3)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_position_embeddings", type=int, default=4096)
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--data_file", type=str, default="train.csv")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class PretrainingDataset(Dataset):
    def __init__(self,
                 input_file,
                 tokenizer,
                 max_encoder_length=512,
                 max_predictions_per_seq=75,
                 masked_lm_prob=0.15,
                 pad_val=0,
                 cls_val=65,
                 sep_val=66,
                 mask_val=67,
                 mask_prob=0.8,
                 random_prob=0.1):
        self.tokenizer = tokenizer
        self.max_encoder_length = max_encoder_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.pad_val = pad_val
        input_file = open(input_file, "r")
        self.lines = input_file.readlines()

        self.vocab_size = tokenizer.vocab_size
        self.word_start_subtoken = np.array([
            tokenizer.vocab.idx_to_token[i][0] == "▁"
            for i in range(self.vocab_size)
        ])
        self.masked_lm_prob = masked_lm_prob
        self.cls_val = cls_val
        self.sep_val = sep_val
        self.mask_val = mask_val
        self.mask_prob = mask_prob
        self.random_prob = random_prob

    def __getitem__(self, index):
        # [input_ids, label]
        line = self.lines[index].rstrip()
        # numpy_mask
        subtokens, masked_lm_positions, masked_lm_ids, masked_lm_weights = self.tokenizer.encode(
            line,
            max_seq_len=self.max_encoder_length,
            max_pred_len=self.max_predictions_per_seq)
        return [
            subtokens, np.zeros_like(subtokens), masked_lm_positions,
            masked_lm_ids, masked_lm_weights, np.zeros(
                [1], dtype="int64")
        ]

    def __len__(self):
        return len(self.lines)


def create_dataloader(input_file, tokenizer, worker_init, batch_size,
                      max_encoder_length):
    pretrain_dataset = PretrainingDataset(input_file, tokenizer,
                                          max_encoder_length)
    train_batch_sampler = paddle.io.BatchSampler(
        pretrain_dataset, batch_size=batch_size, shuffle=False)

    # make masked_lm_positions can be gathered
    def _collate_data(data, stack_fn=Stack()):
        # data: input_ids, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels
        num_fields = len(data[0])
        out = [None] * num_fields

        for i in [0, 1, 5]:
            out[i] = stack_fn([x[i] for x in data])
        batch_size, seq_length = out[0].shape
        size = num_mask = sum(len(x[2]) for x in data)
        # if size % 8 != 0:
        #     size += 8 - (size % 8)
        out[2] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[3] = np.full([size, 1], -1, dtype=np.int64)
        # masked weight
        out[4] = np.full([size], 0, dtype="float32")
        # # Organize as a 1D tensor for gather or use gather_nd
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[2]):
                out[2][mask_token_num] = i * seq_length + pos
                out[3][mask_token_num] = x[3][j]
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        out.append(np.asarray([mask_token_num], dtype=np.float32))
        return out

    dataloader = DataLoader(
        dataset=pretrain_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=_collate_data,
        num_workers=0,
        #worker_init_fn=worker_init,
        return_list=True)
    return dataloader


def do_train(args):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    # get dataloader
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    train_data_loader = create_dataloader(args.data_file, tokenizer,
                                          worker_init, args.batch_size,
                                          args.max_encoder_length)
    logger.info("Dataloader has been created")

    # define model
    model = BigBirdForPretraining(
        BigBirdModel(**BigBirdForPretraining.pretrained_init_configuration[
            args.model_name_or_path]))

    # define metric
    criterion = BigBirdPretrainingCriterion(
        getattr(model, BigBirdForPretraining.base_model_prefix).config[
            "vocab_size"])

    # define optimizer
    lr_scheduler = LinearDecayWithWarmup(
        args.learning_rate,
        args.num_train_steps,
        args.warmup_steps,
        last_epoch=0)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])
    bigbirdConfig = BigBirdModel.pretrained_init_configuration[
        args.model_name_or_path]
    # training
    model.train()
    global_steps = 0
    seed = 0
    np.random.seed(seed)
    for epoch in range(args.epochs):
        if global_steps > args.num_train_steps:
            break
        for step, batch in enumerate(train_data_loader):
            (input_ids, segment_ids, masked_lm_positions, masked_lm_ids,
             masked_lm_weights, next_sentence_labels, masked_lm_scale) = batch
            seq_len = input_ids.shape[1]
            # rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
            #     bigbirdConfig["num_layers"], seq_len, seq_len,
            #     bigbirdConfig["nhead"], bigbirdConfig["block_size"],
            #     bigbirdConfig["window_size"],
            #     bigbirdConfig["num_global_blocks"],
            #     bigbirdConfig["num_rand_blocks"], bigbirdConfig["seed"])
            rand_mask_idx_list = None
            prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                rand_mask_idx_list=rand_mask_idx_list,
                masked_positions=masked_lm_positions)
            loss = criterion(prediction_scores, seq_relationship_score,
                             masked_lm_ids, next_sentence_labels,
                             masked_lm_scale, masked_lm_weights)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            if global_steps % args.logging_steps == 0:
                logger.info(batch)
                logger.info("global step %d, epoch: %d, loss: %f" %
                            (global_steps, epoch, loss))

            global_steps += 1
            if global_steps > args.num_train_steps:
                break
            np.random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)