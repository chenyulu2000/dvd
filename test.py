import os

import torch
import torch.nn as nn

from anatool import AnaArgParser, AnaLogger
from data.dataloader import get_test_dataloader
from test_models.test_base import test_base
from test_models.test_debias_both import test_debias_both
from test_models.test_debias_history import test_debias_history
from test_models.test_debias_image import test_debias_image
from utils.save_experiment import save_experiment


def main(opt, logger):
    test_dataset, test_dataloader = get_test_dataloader(opt=opt, logger=logger)
    shared_word_embed = nn.Embedding(
        num_embeddings=len(test_dataset.vocabulary),
        embedding_dim=opt.word_embedding_size,
        padding_idx=test_dataset.vocabulary.PAD_INDEX
    )

    test_params = {
        'opt': opt, 'logger': logger, 'test_dataloader': test_dataloader,
        'shared_word_embed': shared_word_embed
    }
    if opt.encoder == 'debias':
        if opt.debias_hist and not opt.debias_img:
            test_debias_history(**test_params)
        elif opt.debias_img and not opt.debias_hist:
            test_debias_image(**test_params)
        else:
            test_debias_both(**test_params)
    else:
        test_base(**test_params)


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
            datetime=opt.datetime,
            valset=opt.val_set + opt.split
        )
        opt.exp_path = exp_path
        logger = AnaLogger(exp_saved_path=opt.exp_path)

    torch.cuda.set_device(device=torch.device('cuda', opt.devices[0]))
    logger.info(opt)

    main(opt=opt, logger=logger)

    logger.info(f'Testing done!')
