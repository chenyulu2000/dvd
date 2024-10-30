import json
import os

import torch
from torch.nn import DataParallel
from tqdm import tqdm

from models.models import BaseModel
from utils.checkpointing import load_checkpoint
from utils.metrics import SparseGTMetrics, NDCG, scores_to_ranks


def test_base(opt, logger, test_dataloader, shared_word_embed):
    state_dict = load_checkpoint(checkpoint_pthpath=opt.load_path)

    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    simple_results_path = os.path.join(opt.exp_path, 'results,txt')

    model = BaseModel(opt=opt, shared_word_embed=shared_word_embed)

    model = DataParallel(module=model.cuda(), device_ids=opt.devices)
    model.module.load_state_dict(state_dict['base'])
    model.eval()

    ranks_json = []
    ranks_path = os.path.join(opt.exp_path, 'ranks.json')

    for batch_num, batch in enumerate(tqdm(test_dataloader)):
        for key in batch:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            output = model(batch)
        ranks = scores_to_ranks(output)

        for i in range(len(batch['img_ids'])):
            if opt.split == 'test':
                ranks_json.append({
                    'image_id': batch['img_ids'][i].item(),
                    'round_id': int(batch['num_rounds'][i].item()),
                    'ranks': [rank.item() for rank in ranks[i][batch['num_rounds'][i] - 1]]
                })
            else:
                for j in range(batch["num_rounds"][i]):
                    ranks_json.append({
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks[i][j]],
                    })

        if opt.split == 'val':
            sparse_metrics.observe(predicted_scores=output, target_ranks=batch['ans_ind'])

            if 'gt_relevance' in batch:
                output = output[torch.arange(output.size(0)), batch['round_id'] - 1, :]
                ndcg.observe(predicted_scores=output, target_relevance=batch['gt_relevance'])

    if opt.split == 'val':
        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True, get_last_num_round=True))
        all_metrics.update(ndcg.retrive(reset=True))

        logger.info('Metrics:')
        msg = ''
        for metric_name, metric_value in all_metrics.items():
            logger.info(f'{metric_name}: {metric_value}')
            msg += '%.2f ' % metric_value

        with open(simple_results_path, 'w') as f:
            f.write(f'FG: {msg}\n')

    json.dump(ranks_json, open(ranks_path, 'w'))
