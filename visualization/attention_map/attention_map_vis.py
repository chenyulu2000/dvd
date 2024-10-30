import json

import h5py
import torch
import matplotlib.pyplot as plt
import numpy
from PIL import Image

BACTH_NUM = 2
BATCH_SIZE = 250
IMG_REGION = 36
IMG_ATTENTION_THRES = 0.4
IMG_ATTENTION_MIN = 0

dialog_path = '/data/visdial/visdial_1.0_val.json'
img_features_path = '/data/visdial/visdial_val_features.h5'
origin_img_path = '/data/visdial/VisualDialog_val2018/'
img_scores_fg_path = f'visualization/attention_map/sources/FG_{str(BACTH_NUM).zfill(3)}.npy'
img_scores_bg_path = f'visualization/attention_map/sources/BG_{str(BACTH_NUM).zfill(3)}.npy'
save_path = 'visualization/attention_map/results/imgs/'
sub_save_path = 'visualization/attention_map/results/sub_imgs/'

with open(dialog_path, 'r') as file:
    dialogs = json.load(file)['data']['dialogs']
    dialogs_img_ids = [content['image_id'] for content in dialogs]

with h5py.File(img_features_path, 'r') as file:
    boxes = file['boxes'][:]
    heights = file['h'][:]
    widths = file['w'][:]
    boxes_img_ids = list(map(int, file['image_id']))

img_scores_fg = torch.load(img_scores_fg_path, map_location=torch.device('cpu'))
img_scores_bg = torch.load(img_scores_bg_path, map_location=torch.device('cpu'))

img_scores_fg[img_scores_fg < IMG_ATTENTION_THRES] = IMG_ATTENTION_MIN
img_scores_bg[img_scores_bg < IMG_ATTENTION_THRES] = IMG_ATTENTION_MIN


def reshape_img(x):
    x = x.reshape(BATCH_SIZE, -1, IMG_REGION, IMG_REGION)
    x = x.sum(dim=-1)
    x = x.sum(dim=-1)
    return x.cpu().numpy()


def save_subfig(fig, ax, save_path, fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path + fig_name, bbox_inches=extent)


def visualize_img(id, origin, fg, bg):
    print(id)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes[0, 0].imshow(origin)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fg, cmap='plasma', interpolation='nearest')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(origin)
    axes[1, 0].imshow(fg, alpha=0.6, cmap='plasma', interpolation='nearest')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(origin)
    axes[1, 1].imshow(bg, alpha=0.6, cmap='plasma', interpolation='nearest')
    axes[1, 1].axis('off')

    plt.tight_layout()

    plt.savefig(f'{save_path}img{str(id).zfill(3)}.png', dpi=600)
    save_subfig(fig, axes[0, 0], sub_save_path, f'{str(id).zfill(3)}_origin.png')
    save_subfig(fig, axes[1, 0], sub_save_path, f'{str(id).zfill(3)}_fg.png')
    save_subfig(fig, axes[1, 1], sub_save_path, f'{str(id).zfill(3)}_bg.png')
    # plt.show()


def acc_img_attention(img_id, scores):
    index = boxes_img_ids.index(img_id)

    box = boxes[index]
    hh = heights[index]
    ww = widths[index]

    ts = numpy.zeros([hh, ww])
    for r in range(IMG_REGION):
        x1, y1, x2, y2 = box[r]
        y1, y2 = int(y1), int(y2) - 1
        x1, x2 = int(x1), int(x2) - 1
        if y1 >= y2 or x1 >= x2:
            continue
        try:
            ts[int(y1):int(y2) - 1, int(x1) + 1:int(x2) - 1] += scores[r]
        except:
            pass
    return ts


img_scores_fg = reshape_img(img_scores_fg)
img_scores_bg = reshape_img(img_scores_bg)

for i in range(BATCH_SIZE):
    _img_scores_fg, _img_scores_bg = img_scores_fg[i], img_scores_bg[i]
    img_id = dialogs_img_ids[i + BACTH_NUM * BATCH_SIZE]

    acc_fg = acc_img_attention(img_id, _img_scores_fg)
    acc_bg = acc_img_attention(img_id, _img_scores_bg)

    visualize_img(
        i + BACTH_NUM * BATCH_SIZE,
        numpy.array(Image.open(f'{origin_img_path}VisualDialog_val2018_000000{str(img_id).zfill(6)}.jpg')),
        acc_fg, acc_bg
    )
