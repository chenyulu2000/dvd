import datetime
import itertools
import os.path
import re
from bisect import bisect

import torch.cuda
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from models.encoders.disentanglement import ImageEncoder, QuestionEncoder, HistoryEncoder
from models.models import DebiasModel
from utils.checkpointing import load_checkpoint, CheckpointManager
from utils.loss import ndcg_loss, ce_loss, generalized_ce_loss
from utils.metrics import SparseGTMetrics, NDCG


def solver_debias_both(
        opt, logger, train_dataset, val_dataset, finetune,
        image_encoder_fg, image_encoder_bg,
        question_encoder, history_encoder_fg, history_encoder_bg,
        debias_both_model_fg, debias_both_model_bg
):
    initial_lr = opt.initial_lr_curriculum if finetune else opt.initial_lr
    if torch.cuda.device_count() > 0:
        nodes = torch.cuda.device_count()
    else:
        nodes = 1

    if opt.training_splits == 'trainval':
        iterations = 1 + (len(train_dataset) + len(val_dataset)) // (opt.batch_size * nodes)
    else:
        iterations = 1 + len(train_dataset) // (opt.batch_size * nodes)

    def lr_lambda_fun(current_iteration) -> float:
        current_epoch = float(current_iteration) / iterations
        if current_epoch <= opt.warmup_epochs:
            alpha = current_epoch / float(opt.warmup_epochs)
            return opt.warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(opt.lr_milestones, current_epoch)
            return pow(opt.lr_gamma, idx)

    logger.info(f'Initial learning rate set to: {initial_lr}.')
    optimizer_fg = optim.Adamax(
        list(image_encoder_fg.parameters()) + list(question_encoder.parameters()) +
        list(history_encoder_fg.parameters()) + list(debias_both_model_fg.parameters()),
        lr=initial_lr
    )
    optimizer_bg = optim.Adamax(
        list(image_encoder_bg.parameters()) + list(debias_both_model_bg.parameters()) +
        list(history_encoder_bg.parameters()),
        lr=initial_lr
    )
    scheduler_fg = optim.lr_scheduler.LambdaLR(optimizer=optimizer_fg, lr_lambda=lr_lambda_fun)
    scheduler_bg = optim.lr_scheduler.LambdaLR(optimizer=optimizer_bg, lr_lambda=lr_lambda_fun)
    return optimizer_fg, optimizer_bg, scheduler_fg, scheduler_bg, iterations


def train_debias_both(opt, logger, train_dataloader, train_dataset, val_dataloader, val_dataset,
                      eval_dataloader, shared_word_embed, summary_writer, finetune):
    sparse_metrics_fg, sparse_metrics_bg = SparseGTMetrics(), SparseGTMetrics()
    ndcg_fg, ndcg_bg = NDCG(), NDCG()

    simple_results_path = os.path.join(opt.exp_path, 'results,txt')

    models_dict = {
        'image_encoder_fg': ImageEncoder(opt=opt, disentangle=True),
        'image_encoder_bg': ImageEncoder(opt=opt, disentangle=True),
        'question_encoder': QuestionEncoder(opt=opt, shared_word_embed=shared_word_embed),
        'history_encoder_fg': HistoryEncoder(opt=opt, shared_word_embed=shared_word_embed, disentangle=True),
        'history_encoder_bg': HistoryEncoder(opt=opt, shared_word_embed=shared_word_embed, disentangle=True),
        'debias_both_model_fg': DebiasModel(opt=opt, shared_word_embed=shared_word_embed),
        'debias_both_model_bg': DebiasModel(opt=opt, shared_word_embed=shared_word_embed)
    }

    for model_name, model in models_dict.items():
        models_dict[model_name] = DistributedDataParallel(module=model.cuda(), device_ids=[opt.local_rank])

    optimizer_fg, optimizer_bg, scheduler_fg, scheduler_bg, iterations = solver_debias_both(
        opt=opt,
        logger=logger,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        finetune=finetune,
        **models_dict
    )

    checkpoint_manager = CheckpointManager(
        models=models_dict,
        optimizers={
            'optimizer_fg': optimizer_fg,
            'optimizer_bg': optimizer_bg
        },
        logger=logger,
        opt=opt,
        checkpoint_dirpath=os.path.join(opt.exp_path, 'ckpts'),
    )

    if not finetune and opt.load_path == '':
        start_epoch = 1
    else:
        try:
            file_name = os.path.basename(opt.load_path)
            start_epoch = int(re.findall('\d+', file_name)[-1]) + 1

            state_dict = load_checkpoint(checkpoint_pthpath=opt.load_path)
            checkpoint_manager.update_last_epoch(start_epoch)

            for model_name, model in models_dict.items():
                model.module.load_state_dict(state_dict[model_name])

            if not finetune:
                optimizer_fg.load_state_dict(state_dict['optimizer_fg'])
                optimizer_bg.load_state_dict(state_dict['optimizer_bg'])
        except:
            logger.error('Error in loading checkpoint.')
            start_epoch = 1

    if finetune:
        end_epoch = start_epoch + opt.num_epochs_curriculm
    else:
        end_epoch = opt.num_epochs

    global_iteration_step = (start_epoch - 1) * iterations

    best_val_ndcg_fg, best_val_ndcg_bg = 0.0, 0.0

    running_loss = 0.0
    for epoch in range(start_epoch, end_epoch + 1):
        if opt.training_splits == 'trainval':
            train_dataloader.sampler.set_epoch(epoch=epoch)
            val_dataloader.sampler.set_epoch(epoch=epoch)
            dataloader = itertools.chain(train_dataloader, val_dataloader)
        elif opt.training_splits == 'train':
            train_dataloader.sampler.set_epoch(epoch=epoch)
            dataloader = itertools.chain(train_dataloader)
        else:
            val_dataloader.sampler.set_epoch(epoch=epoch)
            dataloader = itertools.chain(val_dataloader)

        begin_time = datetime.datetime.now()
        logger.info(f'Training for epoch {epoch}.')
        for i, batch in enumerate(tqdm(dataloader)):
            for key in batch:
                batch[key] = batch[key].cuda()
            optimizer_fg.zero_grad()
            optimizer_bg.zero_grad()
            batch_size = batch['img_feat'].size(0)
            img_features_fg, img_features_mask_fg = models_dict.get('image_encoder_fg')(batch['img_feat'])
            img_features_bg, img_features_mask_bg = models_dict.get('image_encoder_bg')(batch['img_feat'])
            ques_features, ques_features_mask, _ = models_dict.get('question_encoder')(batch['ques'], batch['ques_len'])
            hist_features_fg, hist_features_mask_fg, _ = models_dict.get('history_encoder_fg')(
                batch['cap'], batch['cap_len'], batch['hist'], batch['hist_len'])
            hist_features_bg, hist_features_mask_bg, _ = models_dict.get('history_encoder_bg')(
                batch['cap'], batch['cap_len'], batch['hist'], batch['hist_len'])
            output_fg = models_dict.get('debias_both_model_fg')(
                batch_size, img_features_fg, img_features_bg.detach(), ques_features,
                hist_features_fg, hist_features_bg.detach(),
                img_features_mask_fg, img_features_mask_bg, ques_features_mask,
                hist_features_mask_fg, hist_features_mask_bg, batch
            )

            output_bg = models_dict.get('debias_both_model_bg')(
                batch_size, img_features_fg.detach(), img_features_bg, ques_features,
                hist_features_fg.detach(), hist_features_bg,
                img_features_mask_fg, img_features_mask_bg, ques_features_mask,
                hist_features_mask_fg, hist_features_mask_bg, batch
            )

            if finetune:
                target = batch['gt_relevance']
                output_fg = output_fg[torch.arange(output_fg.size(0)), batch['round_id'] - 1, :]
                output_bg = output_bg[torch.arange(output_bg.size(0)), batch['round_id'] - 1, :]
                loss_fg = ndcg_loss(output=output_fg, labels=target)
                loss_bg = ndcg_loss(output=output_bg, labels=target)
                loss_weight = (loss_bg.detach()) / (loss_fg.detach() + loss_bg.detach() + 1e-8)
                loss = loss_weight * loss_fg + opt.debias_weight * loss_bg
            else:
                loss_fg = ce_loss(decoder_type=opt.decoder, batch=batch, output=output_fg)
                loss_bg = ce_loss(decoder_type=opt.decoder, batch=batch, output=output_bg)
                loss_gce_bg = generalized_ce_loss(decoder_type=opt.decoder, batch=batch, output=output_bg, q=opt.gce_q)
                loss_weight = (loss_fg.detach()) / (loss_fg.detach() + loss_bg.detach() + 1e-8)
                loss = loss_weight * loss_fg + opt.debias_weight * loss_gce_bg

            loss.backward()
            optimizer_fg.step()
            optimizer_bg.step()

            if running_loss > 0.0:
                running_loss = 0.95 * running_loss + 0.05 * loss.item()
            else:
                running_loss = loss.item()

            scheduler_fg.step()
            scheduler_bg.step()

            global_iteration_step += 1
            if global_iteration_step % 100 == 0:
                logger.info("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][LR_FG: {:6f}][LR_BG: {:6f}]".format(
                    datetime.datetime.now() - begin_time, epoch,
                    global_iteration_step, running_loss,
                    optimizer_fg.param_groups[0]['lr'],
                    optimizer_bg.param_groups[0]['lr']
                ))
                summary_writer.add_scalar(
                    tag='train/loss',
                    scalar_value=loss,
                    global_step=global_iteration_step
                )
                summary_writer.add_scalar(
                    tag='train/lr_fg',
                    scalar_value=optimizer_fg.param_groups[0]['lr'],
                    global_step=global_iteration_step
                )
                summary_writer.add_scalar(
                    tag='train/lr_bg',
                    scalar_value=optimizer_bg.param_groups[0]['lr'],
                    global_step=global_iteration_step
                )

        torch.cuda.empty_cache()

        if not finetune:
            checkpoint_manager.step(epoch=epoch)
        else:
            logger.info('Validating before checkpointing.')

        if epoch >= opt.validate_epoch and opt.local_rank == 0:
            for model in models_dict.values():
                model.eval()

            logger.info(f'\nValidation after epoch {epoch}:')
            for i, batch in enumerate(tqdm(eval_dataloader)):
                for key in batch:
                    batch[key] = batch[key].cuda()
                with torch.no_grad():
                    batch_size = batch['img_feat'].size(0)
                    img_features_fg, img_features_mask_fg = models_dict.get('image_encoder_fg')(batch['img_feat'])
                    img_features_bg, img_features_mask_bg = models_dict.get('image_encoder_bg')(batch['img_feat'])
                    ques_features, ques_features_mask, _ = models_dict.get('question_encoder')(
                        batch['ques'], batch['ques_len'])
                    hist_features_fg, hist_features_mask_fg, _ = models_dict.get('history_encoder_fg')(
                        batch['cap'], batch['cap_len'], batch['hist'], batch['hist_len'])
                    hist_features_bg, hist_features_mask_bg, _ = models_dict.get('history_encoder_bg')(
                        batch['cap'], batch['cap_len'], batch['hist'], batch['hist_len'])

                    output_fg = models_dict.get('debias_both_model_fg')(
                        batch_size, img_features_fg, img_features_bg, ques_features, hist_features_fg, hist_features_bg,
                        img_features_mask_fg, img_features_mask_bg, ques_features_mask,
                        hist_features_mask_fg, hist_features_mask_bg, batch
                    )
                    output_bg = models_dict.get('debias_both_model_bg')(
                        batch_size, img_features_fg, img_features_bg, ques_features, hist_features_fg, hist_features_bg,
                        img_features_mask_fg, img_features_mask_bg, ques_features_mask,
                        hist_features_mask_fg, hist_features_mask_bg, batch
                    )

                sparse_metrics_fg.observe(predicted_scores=output_fg, target_ranks=batch['ans_ind'])
                sparse_metrics_bg.observe(predicted_scores=output_bg, target_ranks=batch['ans_ind'])

                if 'gt_relevance' in batch:
                    output_fg = output_fg[torch.arange(output_fg.size(0)), batch['round_id'] - 1, :]
                    output_bg = output_bg[torch.arange(output_bg.size(0)), batch['round_id'] - 1, :]
                    ndcg_fg.observe(output_fg, batch['gt_relevance'])
                    ndcg_bg.observe(output_bg, batch['gt_relevance'])

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

            summary_writer.add_scalars(
                main_tag='metrics_fg',
                tag_scalar_dict=all_metrics_fg,
                global_step=global_iteration_step
            )
            summary_writer.add_scalars(
                main_tag='metrics_bg',
                tag_scalar_dict=all_metrics_bg,
                global_step=global_iteration_step
            )

            for model in models_dict.values():
                model.train()

            torch.cuda.empty_cache()

            val_ndcg_fg, val_ndcg_bg = all_metrics_fg['ndcg'], all_metrics_bg['ndcg']

            checkpoint_manager.step()

            if val_ndcg_fg > best_val_ndcg_fg:
                logger.info(f'Best ForeGround model found at epoch {epoch}.')
                best_val_ndcg_fg = val_ndcg_fg
                checkpoint_manager.save_best(ckpt_name='best_fg_ndcg')
            else:
                logger.info(f'Not saving ForeGround model at epoch {epoch}.')

            if val_ndcg_bg > best_val_ndcg_bg:
                logger.info(f'Best BackGround model found at epoch {epoch}.')
                best_val_ndcg_bg = val_ndcg_bg
                checkpoint_manager.save_best(ckpt_name='best_bg_ndcg')
            else:
                logger.info(f'Not saving BackGround model at epoch {epoch}.')
