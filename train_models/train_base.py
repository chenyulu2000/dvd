import datetime
import itertools
import os.path
import re
from bisect import bisect

import torch.cuda
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from models.models import BaseModel
from utils.checkpointing import load_checkpoint, CheckpointManager
from utils.loss import ndcg_loss, ce_loss
from utils.metrics import SparseGTMetrics, NDCG


def solver_base(opt, logger, train_dataset, val_dataset, finetune, model):
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
    optimizer = optim.Adamax(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda_fun)
    return optimizer, scheduler, iterations


def train_base(opt, logger, train_dataloader, train_dataset, val_dataloader, val_dataset,
               eval_dataloader, shared_word_embed, summary_writer, finetune):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    simple_results_path = os.path.join(opt.exp_path, 'results,txt')

    model = BaseModel(opt=opt, shared_word_embed=shared_word_embed)

    model = DistributedDataParallel(module=model.cuda(), device_ids=[opt.local_rank])

    optimizer, scheduler, iterations = solver_base(
        opt=opt,
        logger=logger,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        finetune=finetune,
        model=model
    )

    checkpoint_manager = CheckpointManager(
        models={'base': model},
        optimizers={'optimizer': optimizer},
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

            model.module.load_state_dict(state_dict['base'])

            if not finetune:
                optimizer.load_state_dict(state_dict['optimizer'])
        except:
            logger.error('Error in loading checkpoint.')
            start_epoch = 1

    if finetune:
        end_epoch = start_epoch + opt.num_epochs_curriculm
    else:
        end_epoch = opt.num_epochs

    global_iteration_step = (start_epoch - 1) * iterations

    best_val_ndcg = 0.0

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
            optimizer.zero_grad()

            output = model(batch)

            if finetune:
                target = batch['gt_relevance']
                output = output[
                         torch.arange(output.size(0)),
                         batch['round_id'] - 1,
                         :]
                loss = ndcg_loss(output=output, labels=target)
            else:
                loss = ce_loss(decoder_type=opt.decoder, batch=batch, output=output)

            loss.backward()
            optimizer.step()

            if running_loss > 0.0:
                running_loss = 0.95 * running_loss + 0.05 * loss.item()
            else:
                running_loss = loss.item()

            scheduler.step()

            global_iteration_step += 1
            if global_iteration_step % 100 == 0:
                logger.info("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][LR: {:6f}]".format(
                    datetime.datetime.now() - begin_time, epoch,
                    global_iteration_step, running_loss,
                    optimizer.param_groups[0]['lr']
                ))
                summary_writer.add_scalar(
                    tag='train/loss',
                    scalar_value=loss,
                    global_step=global_iteration_step
                )
                summary_writer.add_scalar(
                    tag='train/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=global_iteration_step
                )

        torch.cuda.empty_cache()

        if not finetune:
            checkpoint_manager.step(epoch=epoch)
        else:
            logger.info('Validating before checkpointing.')

        if epoch >= opt.validate_epoch and opt.local_rank == 0:
            model.eval()

            logger.info(f'\nValidation after epoch {epoch}:')
            for i, batch in enumerate(tqdm(eval_dataloader)):
                for key in batch:
                    batch[key] = batch[key].cuda()
                with torch.no_grad():
                    output = model(batch)

                sparse_metrics.observe(predicted_scores=output, target_ranks=batch['ans_ind'])

                if 'gt_relevance' in batch:
                    output = output[torch.arange(output.size(0)), batch['round_id'] - 1, :]
                    ndcg.observe(output, batch['gt_relevance'])

            all_metrics = {}
            all_metrics.update(sparse_metrics.retrieve(reset=True, get_last_num_round=True))
            all_metrics.update(ndcg.retrive(reset=True))

            logger.info('Metrics:')
            msg = ''
            for metric_name, metric_value in all_metrics.items():
                logger.info(f'{metric_name}: {metric_value}')
                msg += '%.2f ' % metric_value

            with open(simple_results_path, 'a') as f:
                f.write(f'{msg}\n')

            summary_writer.add_scalars(
                main_tag='metrics',
                tag_scalar_dict=all_metrics,
                global_step=global_iteration_step
            )

            model.train()

            torch.cuda.empty_cache()

            val_ndcg = all_metrics['ndcg']

            if val_ndcg > best_val_ndcg:
                logger.info(f'Best model found at epoch {epoch}.')
                best_val_ndcg = val_ndcg
                checkpoint_manager.save_best(ckpt_name='best_ndcg')
            else:
                logger.info(f'Not saving model at epoch {epoch}.')
