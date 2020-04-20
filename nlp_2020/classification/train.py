import os
import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator

from tqdm import tqdm, trange

from nlp_2020.classification.model import TextClassifier
from nlp_2020.classification.tool import build_dataset
from nlp_2020.classification.args import get_args

logger = logging.getLogger(__name__)
writer = SummaryWriter()


def train(args, model: TextClassifier, dataloader, optimizer, scheduler,
          criterion):
    # Build train dataset
    fields, train_dataset = build_dataset(fpath=args.data_dir, mode='train')

    # Build vocab
    ID, CATEGORY, NEWS = fields
    vectors = Vectors(name=args.embed_fpath, cache=args.data_dir)
    # NOTE: use train_dataset to build vocab!
    NEWS.build_vocab(
        train_dataset,
        max_size=args.vocab_max_size,
        vectors=vectors,
        unk_init=torch.nn.init.xavier_normal_,
    )

    # Init embeddings for model
    pretrained_embeddings = NEWS.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # NOTE: since we don't know the embeddings while pretraining embeddings,
    # we have to assign `unknown` and `pad` explicitly.
    UNK_IDX = NEWS.vocab.stoi[NEWS.unk_token]
    PAD_IDX = NEWS.vocab.stoi[NEWS.pad_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(args.embed_size)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(args.embed_size)

    bucket_iterator = BucketIterator(
        train_dataset,
        batch_size=args.train_batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.news),
        device=args.device,
    )

    global_step = 0
    model.zero_grad()
    train_trange = trange(0, args.num_train_epochs, desc="Train epoch")
    for _ in train_trange:
        epoch_iterator = tqdm(bucket_iterator, desc='Training')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            news, news.lengths = batch.news
            category = batch.category
            preds = model(news, news.lengths)

            # TODO: implement criterion
            loss = criterion(preds, category)
            loss.backword()
            # Refer: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Logging
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/lr', scheduler.get_last_lr(), global_step)

            # NOTE:Evaluate
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                results = evaluate(args, model)
                for key, value in results.items():
                    writer.add_scalar("Eval/{}".format(key), value,
                                      global_step)

            # NOTE: save model
            # TODO: finish save_model()
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, model, optimizer, scheduler, global_step)

    writer.close()


def evaluate(args, model):
    pass


def main():
    args = get_args()
    pass

    # Init