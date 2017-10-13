
# CLEVR models

This repo aims at reproducing the results of CLEVR from the following paper:
- Learning Visual Reasoning Without Strong Priors [1] https://arxiv.org/abs/1707.03017
- FiLM: Visual Reasoning with a General Conditioning Layer [2] https://arxiv.org/abs/1709.07871

and unpublished results from:
-  Modulating early visual processing by language [1] https://arxiv.org/abs/1707.00683

The code was equally developed by Florian Strub (University of Lille) and Harm de Vries (University of Montreal)

The project is part of the CHISTERA - IGLU Project.

PS: more results to come...

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
    * [Download](#Download)
    * [Requirements](#requirements)
    * [File architecture](#file-architecture)
    * [Data](#data)
    * [Pretrained models](#pretrained-models)
* [Reproducing results](#reproducing-results)
    * [Process Data](#data)
    * [Train Model](#train-model)
* [FAQ](#faq)
* [Citation](#citation)

## Introduction

We introduce a new CLEVR Baseline based on FiLM layers and Conditional Batch Normalization technique.

## Installation


### Download

Our code has internal dependencies called submodules. To properly clone the repository, please use the following git command:\

```
git clone --recursive https://github.com/GuessWhatGame/clevr.git
```

### Requirements

The code works on both python 2 and 3. It relies on the tensorflow python API.
It requires the following python packages:

```
pip install \
    tensorflow-gpu \
    nltk \
    tqdm
```


### File architecture
In the following, we assume that the following file/folder architecture is respected:

```
clevr
├── config         # store the configuration file to create/train models
|   └── clevr
|
├── out            # store the output experiments (checkpoint, logs etc.)
|   └── clevr
|
├── data          # contains the CLEVR data
|
└── src            # source files
```

To complete the git-clone file arhictecture, you can do:

```
cd guesswhat
mkdir data;
mkdir out; mkdir out/clevr
```

Of course, one is free to change this file architecture!

### Data
CLEVR relies on the CLEVR dataset: http://cs.stanford.edu/people/jcjohns/clevr/

To download the CLEVR dataset please use wget:
```
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip -P data/
```


### Pretrained networks

TO COME

## Reproducing results

To launch the experiments in the local directory, you first have to set the pyhton path:
```
export PYTHONPATH=src:${PYTHONPATH}
```
Note that you can also directly execute the experiments in the source folder.

### Process Data

Before starting the training, one needs to create a dictionary

#### Extract image features
Following the original papers, we are going to extract fc8 features from the coco images by using a VGG-16 network.

First, one need to download the resnet pretrained network (152) provided by [slim-tensorflow](https://github.com/tensorflow/models/tree/master/slim):

```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -P data/
tar zxvf data/vgg_16_2016_08_28.tar.gz -C data/
```

GuessWhat?! requires to both computes the image features from the full picture
To do so, you need to use the pythn script guesswhat/src/guesswhat/preprocess_data/extract_img_features.py .
```
array=( img crop )
for mode in "${array[@]}"; do
   python src/guesswhat/preprocess_data/extract_img_features.py \
     -image_dir data/img/raw \
     -data_dir data \
     -data_out data \
     -network vgg \
     -ckpt data/vgg_16.ckpt \
     -feature_name fc8 \
     -mode $mode
done
```

Noticeably, one can also extract VGG-fc7 or Resnet150-block4 features. Please follow the script documentation for more advanced setting.




#### Create dictionary

To create the CLEVR dictionary, you need to use the python script clevr/src/clevr/preprocess_data/create_dico.py .

```
python src/clevr/preprocess_data/create_dictionary.py -data_dir data -dict_file dict.json
```



### Train Model
To train the network, you need to select/configure the kind of neural architecure you want.
To do so, you have update the file config/clevr/config.json

Once the config file is set, you can launch the training step:
```
python src/clevr/train/train_clevr.py \
   -data_dir data \
   -img_dir data/img \
   -config config/clevr/raw.json \
   -exp_dir out/clevr \
   -no_thread 2
```

After training, we obtained the following results:

TBD



## FAQ

 - When I start a python script, I have the following message: ImportError: No module named generic.data_provider.iterator (or equivalent module). It is likely that your python path is not correctly set. Add the "src" folder to your python path (PYTHONPATH=src)


## Citation


```
@inproceedings{perez2017learning,
  title={Learning Visual Reasoning Without Strong Priors},
  author={Perez, Ethan and de Vries, Harm and Strub, Florian and Dumoulin, Vincent and Courville, Aaron},
  booktitle={ICML Machine Learning in Speech and Language Processing Workshop},
  year={2017}
}

@article{perez2017film,
  title={FiLM: Visual Reasoning with a General Conditioning Layer},
  author={Perez, Ethan and Strub, Florian and de Vries, Harm and Dumoulin, Vincent and Courville, Aaron},
  journal={arXiv preprint arXiv:1709.07871},
  year={2017}
}

@inproceedings{guesswhat_game,
author = {Harm de Vries and Florian Strub and J\'er\'emie Mary and Hugo Larochelle and Olivier Pietquin and Aaron C. Courville},
title = {Modulating early visual processing by language},
booktitle = {Advances in Neural Information Processing Systems 30},
year = {2017}
url = {https://arxiv.org/abs/1707.00683}
}
```


## Acknowledgement
 - SequeL Team
 - Mila Team






