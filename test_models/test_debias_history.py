import json
import os

import torch
from torch.nn import DataParallel
from tqdm import tqdm

from models.encoders.disentanglement import ImageEncoder, QuestionEncoder, HistoryEncoder
from models.models import DebiasHistoryModel
from utils.checkpointing import load_checkpoint
from utils.metrics import SparseGTMetrics, NDCG, scores_to_ranks


def test_debias_history(opt, logger, test_dataloader, shared_word_embed):
    state_dict = load_checkpoint(checkpoint_pthpath=opt.load_path)

    sparse_metrics_fg, sparse_metrics_bg = SparseGTMetrics(), SparseGTMetrics()
    ndcg_fg, ndcg_bg = NDCG(), NDCG()

    simple_results_path = os.path.join(opt.exp_path, 'results,txt')

    models_dict = {
        'image_encoder': ImageEncoder(opt=opt),
        'question_encoder': QuestionEncoder(opt=opt, shared_word_embed=shared_word_embed),
        'history_encoder_fg': HistoryEncoder(opt=opt, shared_word_embed=shared_word_embed, disentangle=True),
        'history_encoder_bg': HistoryEncoder(opt=opt, shared_word_embed=shared_word_embed, disentangle=True),
        'debias_history_model_fg': DebiasHistoryModel(opt=opt, shared_word_embed=shared_word_embed),
        'debias_history_model_bg': DebiasHistoryModel(opt=opt, shared_word_embed=shared_word_embed)
    }
    for model_name, model in models_dict.items():
        models_dict[model_name] = DataParallel(module=model.cuda(), device_ids=opt.devices)
        models_dict[model_name].module.load_state_dict(state_dict[model_name])
        models_dict[model_name].eval()

    ranks_json_fg, ranks_json_bg = [], []
    ranks_fg_path = os.path.join(opt.exp_path, 'ranks_fg.json')
    ranks_bg_path = os.path.join(opt.exp_path, 'ranks_bg.json')

    for batch_num, batch in enumerate(tqdm(test_dataloader)):
        for key in batch:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            batch_size = batch['img_feat'].size(0)
            img_features, img_features_mask = models_dict.get('image_encoder')(batch['img_feat'])
            ques_features, ques_features_mask, _ = models_dict.get('question_encoder')(
                batch['ques'], batch['ques_len'])
            hist_features_fg, hist_features_mask_fg, _ = models_dict.get('history_encoder_fg')(
                batch['cap'], batch['cap_len'], batch['hist'], batch['hist_len'])
            hist_features_bg, hist_features_mask_bg, _ = models_dict.get('history_encoder_bg')(
                batch['cap'], batch['cap_len'], batch['hist'], batch['hist_len'])

            output_fg = models_dict.get('debias_history_model_fg')(
                batch_size, img_features, ques_features, hist_features_fg, hist_features_bg,
                img_features_mask, ques_features_mask,
                hist_features_mask_fg, hist_features_mask_bg, batch
            )
            output_bg = models_dict.get('debias_history_model_bg')(
                batch_size, img_features, ques_features, hist_features_fg, hist_features_bg,
                img_features_mask, ques_features_mask,
                hist_features_mask_fg, hist_features_mask_bg, batch
            )

        ranks_fg, ranks_bg = scores_to_ranks(output_fg), scores_to_ranks(output_bg)

        for i in range(len(batch['img_ids'])):
            if opt.split == 'test':
                ranks_json_fg.append({
                    'image_id': batch['img_ids'][i].item(),
                    'round_id': int(batch['num_rounds'][i].item()),
                    'ranks': [rank.item() for rank in ranks_fg[i][batch['num_rounds'][i] - 1]]
                })
                ranks_json_bg.append({
                    'image_id': batch['img_ids'][i].item(),
                    'round_id': int(batch['num_rounds'][i].item()),
                    'ranks': [rank.item() for rank in ranks_bg[i][batch['num_rounds'][i] - 1]]
                })
            else:
                for j in range(batch["num_rounds"][i]):
                    ranks_json_fg.append({
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks_fg[i][j]],
                    })
                for j in range(batch["num_rounds"][i]):
                    ranks_json_bg.append({
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks_bg[i][j]],
                    })

        if opt.split == 'val':
            sparse_metrics_fg.observe(predicted_scores=output_fg, target_ranks=batch['ans_ind'])
            sparse_metrics_bg.observe(predicted_scores=output_bg, target_ranks=batch['ans_ind'])

            if 'gt_relevance' in batch:
                output_fg = output_fg[torch.arange(output_fg.size(0)), batch['round_id'] - 1, :]
                output_bg = output_bg[torch.arange(output_bg.size(0)), batch['round_id'] - 1, :]
                ndcg_fg.observe(predicted_scores=output_fg, target_relevance=batch['gt_relevance'])
                ndcg_bg.observe(predicted_scores=output_bg, target_relevance=batch['gt_relevance'])

    if opt.split == 'val':
        all_metrics_fg, all_metrics_bg = {}, {}
        all_metrics_fg.update(sparse_metrics_fg.retrieve(reset=True, get_last_num_round=True))
        all_metrics_bg.update(sparse_metrics_bg.retrieve(reset=True, get_last_num_round=True))
        all_metrics_fg.update(ndcg_fg.retrive(reset=True))
        all_metrics_bg.update(ndcg_bg.retrive(reset=True))

        logger.info('ForeGround Metrics:')
        fg_msg = ''
        for metric_name, metric_value in all_metrics_fg.items():
            logger.info(f'{metric_name}: {metric_value}')
            fg_msg += '%.2f ' % metric_value

        logger.info('BackGround Metrics:')
        for metric_name, metric_value in all_metrics_bg.items():
            logger.info(f'{metric_name}: {metric_value}')

        with open(simple_results_path, 'a') as f:
            f.write(f'FG: {fg_msg}\n')

    json.dump(ranks_json_fg, open(ranks_fg_path, 'w'))
    json.dump(ranks_json_bg, open(ranks_bg_path, 'w'))
