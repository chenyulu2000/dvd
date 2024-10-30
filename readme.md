## DVD (Debiased Visual Dialog Model)

<div><img src="assets/overview.jpg" alt=""></div>

## Credits

This repository is build upon [visdial-challenge-starter-pytorch](https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch) (Das et al.) and [visdial_conv](https://github.com/shubhamagarwal92/visdial_conv) (Agarwal et al.). We
express our sincere gratitude to the researchers for providing their code, which has been instrumental in the
development of this project.

## Environment Configuration

```shell
conda create -n dvd python=3.7.13
conda activate dvd
pip install -r requirements.txt
```

```shell
python -c "import nltk; nltk.download('all')"
```

## Data Preparation
Download the following files and place them in the specified directory according to [dataset.yaml](configs/dataset.yaml).
<table>
	<tr>
	    <th>Dataset</th>
	    <th>File</th>
	    <th>Source</th>  
	</tr >
	<tr >
	    <td rowspan="10">Visdial v1.0</td>
	    <td style="text-align:center">features_faster_rcnn_x101_train.h5</td>
	    <td rowspan="4" style="text-align:center"><a href="https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch">visdial-challenge-starter-pytorch</a>
 (Das et al.)</td>
	</tr>
    <tr>
	    <td style="text-align:center">features_faster_rcnn_x101_val.h5</td>
	</tr>
    <tr>
	    <td style="text-align:center">features_faster_rcnn_x101_test.h5</td>
	</tr>
	<tr>
	    <td style="text-align:center">visdial_1.0_word_counts_train.json</td>
	</tr>
    <tr>
	    <td style="text-align:center">glove.npy</td>
        <td style="text-align:center"><a href="https://github.com/simpleshinobu/visdial-principles">visdial-principles</a>(Qi et al.)</td>
	</tr>
    <tr>
	    <td style="text-align:center">visdial_1.0_train.json</td>
        <td rowspan="5" style="text-align:center"><a href="https://visualdialog.org/data">visual dialog website</a></td>
	</tr>
	<tr>
	    <td style="text-align:center">visdial_1.0_val.json</td>
	</tr>
    <tr>
	    <td style="text-align:center">visdial_1.0_test.json</td>
	</tr>
    <tr>
	    <td style="text-align:center">visdial_1.0_train_dense_annotations.json (Optional)</td>
	</tr>
    <tr>
	    <td style="text-align:center">visdial_1.0_val_dense_annotations.json</td>
	</tr>
    <tr >
	    <td rowspan="2" style="text-align:center">VisdialConv</td>
	    <td style="text-align:center">visdial_1.0_val_crowdsourced.json</td>
	    <td rowspan="4" style="text-align:center"><a href="https://github.com/shubhamagarwal92/visdial_conv">visdial_conv</a> (Agarwal et al.)</td>
	</tr>
<tr>
	    <td style="text-align:center">visdial_1.0_val_dense_annotations_crowdsourced.json</td>
	</tr>
<tr >
	    <td rowspan="2" style="text-align:center">VisPro</td>
	    <td style="text-align:center">visdial_1.0_val_vispro.json</td>
	</tr>
    <tr>
	    <td style="text-align:center">visdial_1.0_val_dense_annotations_vispro.json</td>
	</tr>
</table>


## Project Structure
The organizational structure of our code is outlined below:
- [train.py](train.py) and [test.py](test.py) -- serve as the entry points for training and testing, respectively, and are invoked by shell scripts.
- [train_models](train_models) and [test_models](test_models) -- contain details pertaining to the training and testing processes for both the dvd and ablation models.
- [scripts](scripts) -- houses all shell scripts.
- [configs](configs) -- encompasses all configurations defined for the project.
- [data](data) -- includes the dataset reader and vocabulary definitions.
- [anatool](anatool) and [utils](utils) -- comprise the implementation of tools such as logger, argparser, checkpoint_manager, etc.
- [exps](exps) -- serves as the repository for storing all experimental data.
- [visualization](visualization) -- contains components related to the visualization of attention maps, learning rate analyses, case studies, etc.
## Training Phase

We use GeForce RTX 3090 and Distributed Data Parallel to train DVD, and the correspondence between model parameters and the number of GPUs and batch size used during training is as follows:

<table>
    <tr>
        <th style="text-align:center">GPUs</th>
        <th style="text-align:center">Batch Size</th>
        <th style="text-align:center">Batch Size per GPU</th>
        <th style="text-align:center">Layers</th>
        <th style="text-align:center">Heads</th>
    </tr>
    <tr>
        <td rowspan="6" style="text-align:center">1</td>
        <td rowspan="12" style="text-align:center">16</td>
        <td rowspan="6" style="text-align:center">16</td>
        <td rowspan="3" style="text-align:center">1</td>
        <td style="text-align:center">2</td>
    </tr>
    <tr>
        <td style="text-align:center">4</td>
    </tr>
    <tr>
        <td style="text-align:center">8</td>
    </tr>
    <tr>
        <td rowspan="3" style="text-align:center">2</td>
        <td style="text-align:center">2</td>
    </tr>
    <tr>
        <td style="text-align:center">4</td>
    </tr>
    <tr>
        <td style="text-align:center">8</td>
    </tr>
    <tr>
        <td rowspan="6" style="text-align:center">2</td>
        <td rowspan="6" style="text-align:center">8</td>
        <td rowspan="3" style="text-align:center">4</td>
        <td style="text-align:center">2</td>
    </tr>
    <tr>
        <td style="text-align:center">4</td>
    </tr>
    <tr>
        <td style="text-align:center">8</td>
    </tr>
    <tr>
        <td rowspan="3" style="text-align:center">6</td>
        <td style="text-align:center">2</td>
    </tr>
    <tr>
        <td style="text-align:center">4</td>
    </tr>
    <tr>
        <td style="text-align:center">8</td>
    </tr>
</table>

<table>
    <tr>
        <th style="text-align:center">GPUs</th>
        <th style="text-align:center">Batch Size</th>
        <th style="text-align:center">Batch Size per GPU</th>
        <th style="text-align:center">Layers</th>
        <th style="text-align:center">Heads</th>
    </tr>
    <tr>
        <td rowspan="3" style="text-align:center">1</td>
        <td style="text-align:center">4</td>
        <td style="text-align:center">4</td>
        <td rowspan="5" style="text-align:center">2</td>
        <td rowspan="5" style="text-align:center">4</td>
    </tr>
    <tr>
        <td style="text-align:center">8</td>
        <td style="text-align:center">8</td>
    </tr>
    <tr>
        <td style="text-align:center">16</td>
        <td style="text-align:center">16</td>
    </tr>
    <tr>
        <td style="text-align:center">2</td>
        <td style="text-align:center">32</td>
        <td style="text-align:center">16</td>
    </tr>
    <tr>
        <td style="text-align:center">4</td>
        <td style="text-align:center">64</td>
        <td style="text-align:center">16</td>
    </tr>
</table>

We provide a debug mode that uses minimal data and does not create a separate experiment folder. To activate this mode, execute the following shell command:
```shell
bash -i scripts/debug_train.sh
```
For regular training, the following command is used:
```shell
bash -i scripts/train.sh
```
The training logs and checkpoints will be saved in directory *exps/exp_name*.
## Test Phase

```shell
bash -i scripts/test.sh
```

The testing logs and checkpoints will be saved in directory *exps/exp_name*.
To get the results generated by [EvalAI](https://eval.ai/web/challenges/challenge-page/518/submission), please submit the file *exps/exp_name/ranks_fg.json*.

## Visualization (Optional)
### Attention Maps
1. Please refer to the
repository [Faster-R-CNN-with-model-pretrained-on-Visual-Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome) and utilize the pre-trained model to
 generate 2048-d image features with bounding boxes.
2. Set the value of **only_attention** in [configs/test.yaml](configs/test.yaml) to **True**, then run [scripts/test.sh](scripts/test.sh) and placing the resulting attention score file in [visualization/attention_map/sources](visualization/attention_map/sources).
3. Run the following script:
```shell
bash -i visualization/attention_map_vis.sh
```
