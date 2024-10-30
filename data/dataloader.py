from torch.utils.data import DistributedSampler, DataLoader

from data.dataset import VisDialDataset


def get_dataloader(opt, logger, finetune=False):
    train_dataset = VisDialDataset(
        opt=opt,
        logger=logger,
        dialogs_json_path=opt.train_json,
        dense_annotations_json_path=opt.train_dense_json,
        finetune=finetune,
        debug=opt.debug,
        in_memory=opt.pin_memory,
        return_options=True if opt.decoder == 'disc' else False,
        add_boundary_toks=False if opt.decoder == 'disc' else True,
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        sampler=train_sampler
    )
    val_dataset = VisDialDataset(
        opt=opt,
        logger=logger,
        dialogs_json_path=opt.val_json,
        dense_annotations_json_path=opt.val_dense_json,
        finetune=finetune,
        debug=opt.debug,
        in_memory=opt.pin_memory,
        return_options=True,
        add_boundary_toks=False if opt.decoder == 'disc' else True
    )
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        sampler=val_sampler
    )
    eval_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.val_batch_size if opt.decoder == 'disc' else 5,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        shuffle=False
    )
    return train_dataloader, val_dataloader, train_dataset, val_dataset, eval_dataloader


def get_test_dataloader(opt, logger):
    if opt.split == 'val':
        test_dataset = VisDialDataset(
            opt=opt,
            logger=logger,
            dialogs_json_path=opt.val_json,
            dense_annotations_json_path=opt.val_dense_json,
            debug=opt.debug,
            in_memory=opt.pin_memory,
            return_options=True,
            add_boundary_toks=False if opt.decoder == 'disc' else True
        )
    else:
        test_dataset = VisDialDataset(
            opt=opt,
            logger=logger,
            dialogs_json_path=opt.test_json,
            debug=opt.debug,
            in_memory=opt.pin_memory,
            return_options=True,
            add_boundary_toks=False if opt.decoder == 'disc' else True
        )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch_size if opt.decoder == 'disc' else 5,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        shuffle=False
    )
    return test_dataset, test_dataloader
