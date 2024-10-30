import os
import shutil


def save_experiment(phase, encoder, decoder, fusion_layer, fusion_head, debias_img,
                    debias_hist, datetime, weight='', gce_q='', valset=''):
    if debias_img and debias_hist:
        debias = 'debias_both'
    elif debias_img and not debias_hist:
        debias = 'debias_image'
    elif not debias_img and debias_hist:
        debias = 'debias_history'
    else:
        debias = 'base'
    if phase == 'test' or debias == 'base':
        weight, gce_q = '', ''
    exp_path = f'exps/{phase}_{encoder}_{decoder}_{fusion_layer}{fusion_head}_{debias}_{str(weight)}_{str(gce_q)}_{valset}_{datetime}'
    try:
        os.makedirs(exp_path)
        src_dirs = ['models', 'utils', 'data']
        src_files = ['configs/dataset.yaml', 'configs/fusion.yaml']
        if phase == 'train':
            src_dirs.append('train_models')
            src_files.extend(['scripts/train.sh', 'configs/train.yaml', 'train.py'])
        else:
            src_dirs.append('test_models')
            src_files.extend(['scripts/test.sh', 'configs/test.yaml', 'test.py'])

        for src_dir in src_dirs:
            shutil.copytree(src_dir, os.path.join(exp_path, src_dir))
        for src_file in src_files:
            shutil.copy(src_file, exp_path)
    except:
        pass
    return exp_path
