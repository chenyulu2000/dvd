import os.path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.backends import cudnn

from anatool import AnaArgParser, AnaLogger
from data.dataloader import get_dataloader
from train_models.train_base import train_base
from train_models.train_debias_both import train_debias_both
from train_models.train_debias_history import train_debias_history
from train_models.train_debias_image import train_debias_image
from utils.save_experiment import save_experiment


def train(opt, logger, finetune=False):
    train_dataloader, val_dataloader, train_dataset, val_dataset, eval_dataloader = get_dataloader(
        opt=opt,
        logger=logger,
        finetune=finetune
    )

    shared_word_embed = nn.Embedding(
        num_embeddings=len(train_dataset.vocabulary),
        embedding_dim=opt.word_embedding_size,
        padding_idx=train_dataset.vocabulary.PAD_INDEX
    )

    summary_writer = SummaryWriter(log_dir=opt.exp_path)

    if opt.glove_npy != '' and os.path.exists(opt.glove_npy):
        shared_word_embed.weight.data = torch.from_numpy(np.load(opt.glove_npy))
        logger.info(f'Loaded glove vectors from: {opt.glove_npy}')

    train_params = {
        'opt': opt, 'logger': logger, 'train_dataloader': train_dataloader, 'train_dataset': train_dataset,
        'val_dataloader': val_dataloader, 'val_dataset': val_dataset, 'eval_dataloader': eval_dataloader,
        'shared_word_embed': shared_word_embed, 'summary_writer': summary_writer, 'finetune': finetune
    }

    if opt.encoder == 'debias':
        if opt.debias_hist and not opt.debias_img:
            train_debias_history(**train_params)
        elif opt.debias_img and not opt.debias_hist:
            train_debias_image(**train_params)
        else:
            train_debias_both(**train_params)
    else:
        train_base(**train_params)


def main(opt, logger):
    if opt.phase in ['train', 'both']:
        logger.info('Starting training.')
        train(opt=opt, logger=logger, finetune=False)
    if opt.phase in ['finetune', 'both']:
        if not os.path.exists(opt.load_path):
            logger.error('Please provide a model path before fine-tuning.')
            raise FileNotFoundError
        logger.info('Starting finetuning.')
        train(opt=opt, logger=logger, finetune=True)


if __name__ == '__main__':
    opt = AnaArgParser().cfg
    if opt.debug:
        opt.exp_path = 'exps/debug'
        if not os.path.exists(opt.exp_path):
            os.mkdir(opt.exp_path)
        logger = AnaLogger()
    else:
        exp_path = save_experiment(
            phase=opt.phase,
            encoder=opt.encoder,
            decoder=opt.decoder,
            fusion_layer=opt.fusion_layer,
            fusion_head=opt.fusion_multi_head,
            debias_img=opt.debias_img,
            debias_hist=opt.debias_hist,
            weight=opt.debias_weight,
            gce_q=opt.gce_q,
            datetime=opt.datetime
        )
        opt.exp_path = exp_path
        logger = AnaLogger(exp_saved_path=opt.exp_path)
    if opt.local_rank < 1:
        logger.info(opt)

    cudnn.benchmark = True

    torch.cuda.set_device(device=opt.local_rank)
    dist.init_process_group(backend='nccl')

    main(opt=opt, logger=logger)

    if opt.local_rank < 1:
        logger.info(f'Training done!')
