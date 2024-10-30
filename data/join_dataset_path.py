import os


def join_dataset_path(opt, val_set='visdial'):
    data_dir = opt.data_dir
    splits = ['train', 'val', 'test']

    if val_set == 'vispro':
        opt[f'val_json'] = opt[f'val_json_vispro']
        opt[f'val_dense_json'] = opt[f'val_dense_json_vispro']
    elif val_set == 'visdialconv':
        opt[f'val_json'] = opt[f'val_json_visdialconv']
        opt[f'val_dense_json'] = opt[f'val_dense_json_visdialconv']

    for split in splits:
        opt[f'image_features_{split}_h5'] = os.path.join(data_dir, opt[f'image_features_{split}_h5'])
        opt[f'{split}_json'] = os.path.join(data_dir, opt[f'{split}_json'])
        if split != 'test':
            opt[f'{split}_dense_json'] = os.path.join(data_dir, opt[f'{split}_dense_json'])
    opt.word_counts_json = os.path.join(
        data_dir,
        opt.word_counts_json
    )
    opt.glove_npy = os.path.join(
        data_dir,
        opt.glove_npy
    )
    return opt
