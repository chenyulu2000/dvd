import json
import os.path

import numpy
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

source_file_path = '/data/visdial/visdial_1.0_val.json'
origin_img_path = '/data/visdial/VisualDialog_val2018'

with open(source_file_path) as f:
    data = json.load(f)['data']

questions = data['questions']
answers = data['answers']
dialogs = data['dialogs']

dialogs_dict = {}
for item in dialogs:
    dialogs_dict[item['image_id']] = item

ranks_file_path_base = 'visualization/case_analysis/ranks/base.json'
ranks_file_path_debias = 'visualization/case_analysis/ranks/debias.json'

with open(ranks_file_path_base) as f:
    ranks_base = json.load(f)

with open(ranks_file_path_debias) as f:
    ranks_debias = json.load(f)

print(len(ranks_base))
M = 1000
round_idx = 8

ranks_base = ranks_base[round_idx::10]
ranks_debias = ranks_debias[round_idx::10]

debias_right_imgs_save_path = 'visualization/case_analysis/results/debias_right/images'
both_right_imgs_save_path = 'visualization/case_analysis/results/both_right/images'
base_right_imgs_save_path = 'visualization/case_analysis/results/base_right/images'
debias_right_cmp_path = 'visualization/case_analysis/results/debias_right/cmp'
both_right_cmp_path = 'visualization/case_analysis/results/both_right/cmp'
base_right_cmp_path = 'visualization/case_analysis/results/base_right/cmp'

if not os.path.exists(debias_right_imgs_save_path):
    os.makedirs(debias_right_imgs_save_path)
    os.makedirs(both_right_imgs_save_path)
    os.makedirs(base_right_imgs_save_path)
    os.makedirs(debias_right_cmp_path)
    os.makedirs(both_right_cmp_path)
    os.makedirs(base_right_cmp_path)


def save_subfig(fig, ax, save_path, fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_path, fig_name), bbox_inches=extent)


def visualize(id, img, text, corrections):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].axis('off')

    axes[1].imshow(text)
    axes[1].axis('off')

    plt.tight_layout()
    if corrections == (True, True):
        img_path = both_right_imgs_save_path
        cmp_path = both_right_cmp_path
    elif corrections == (True, False):
        img_path = debias_right_imgs_save_path
        cmp_path = debias_right_cmp_path
    elif corrections == (False, True):
        img_path = base_right_imgs_save_path
        cmp_path = base_right_cmp_path
    else:
        raise Exception
    plt.savefig(f'{cmp_path}/{id}.png', dpi=600)
    save_subfig(fig, axes[0], img_path, f'{id}.png')


for i in range(len(ranks_base)):
    image_id = ranks_base[i]['image_id']
    top1_index_base = ranks_base[i]['ranks'][0]
    top1_index_debias = ranks_debias[i]['ranks'][0]
    one_dialogs = dialogs_dict[image_id]
    dialog = one_dialogs['dialog']
    gt_index = dialog[round_idx]['gt_index']

    if M == 0:
        break

    if gt_index == top1_index_debias or gt_index == top1_index_base:
        M -= 1
        caption = one_dialogs['caption']
        question_0 = questions[dialog[0]['question']]
        answer_0 = answers[dialog[0]['answer']]
        question_1 = questions[dialog[1]['question']]
        answer_1 = answers[dialog[1]['answer']]

        question = questions[dialog[round_idx]['question']]
        gt_answer = answers[dialog[round_idx]['answer_options'][gt_index]]
        base_answer = answers[dialog[round_idx]['answer_options'][top1_index_base]]
        debias_answer = answers[dialog[round_idx]['answer_options'][top1_index_debias]]

        msg = 'C: ' + caption + '\n' + \
              'Q0: ' + question_0 + '\n' + \
              'A0: ' + answer_0 + '\n' + \
              'Q1: ' + question_1 + '\n' + \
              'A1:' + answer_1 + '\n' + \
              '\n' + \
              'Q:' + question + '\n' + \
              'GT: ' + gt_answer + '\n' + \
              'Base: ' + base_answer + '\n' + \
              'Debias: ' + debias_answer

        width = 2000
        height = 1000
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('visualization/case_analysis/UbuntuMono-R.ttf', 40)
        text_width, text_height = draw.textsize(msg, font)
        y = (height - text_height) // 2
        draw.text((0, y), msg, fill='black', font=font)
        image.save('visualization/case_analysis/tmp.png')
        visualize(
            i,
            numpy.array(Image.open(f'{origin_img_path}/VisualDialog_val2018_000000{str(image_id).zfill(6)}.jpg')),
            numpy.array(Image.open('visualization/case_analysis/tmp.png')),
            (gt_index == top1_index_debias, gt_index == top1_index_base)
        )
