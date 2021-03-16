# Empathy in Text-based Mental Health Support
This repository contains codes and dataset access instructions for the CSE 517: NLP class project. We reproduced results from the 2020 EMNLP paper [A Computational Approach to Understanding Empathy Expressed in
Text-Based Mental Health Support](https://arxiv.org/pdf/2009.08441).

## Introduction

The paper presents a computational approach to understanding how empathy is expressed in online mental health platforms. Sharma et al. collected and shared a corpus of 10k (post, response) pairs annotated using an empathy framework with supporting evidence for annotations (rationales). They used a multi-task RoBERTa-based bi-encoder model for identifying empathy in conversations and extracting rationales underlying its predictions. Their experiments demonstrate that their approach can effectively identify empathic conversations. 

We have replicated their studies, and extended its applications to two new contexts: medical and political discussions. We also experimented with different hyperparameters and training data batch sizes.

For a quick overview of the original project, check out [bdata.uw.edu/empathy](http://bdata.uw.edu/empathy/). For a detailed description, see the [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441).

## Quickstart

### 1. Prerequisites

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```

If running on a CSE machine (we recommend nlpg01), we recommend using a Python virtual environment, as importing some of the modules may not be permitted. Here, the virtual environment is called v. For more information, see [the Python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

```
$ python3 -m pip install --user virtualenv
$ python3 -m venv v
$ source v/bin/activate
```

### 2. Prepare dataset
A sample raw input data file is available in [dataset/sample_input_ER.csv](dataset/sample_input_ER.csv). This file (and other raw input files in the [dataset](dataset) folder) can be converted into a format that is recognized by the model using with following command:

```
$ python3 src/process_data.py --input_path dataset/sample_input_ER.csv --output_path dataset/sample_input_model_ER.csv
```

To process the medical dataset, run the following command to convert the .txt files from the medical data to a .csv that the model will recognize. The input should be the location of the .txt files that you downloaded (see Dataset Access Instructions for link):
```
$ python3 src/med_process_data.py --input_path local1/zsteineh/med_dataset/ --output_path local1/zsteineh/med_dataset/test_med_data.csv
```
Note, the test_med_data.csv file is already processed and available at local1/zsteineh/med_dataset/ on nlpg01. We do not provide it in GitHub as the file is too large.

### 3. Training the model
For training our model on the sample input data, run the following command:
```
$ python3 src/train.py \
	--train_path=dataset/sample_input_model_ER.csv \
	--lr=2e-5 \
	--batch_size=32 \
	--lambda_EI=1.0 \
	--lambda_RE=0.5 \
	--save_model \
	--save_model_path=output/sample.pth
```

### 4. Results 4.3.4 - 4.3.5: Testing the model 
#### Political dataset
The political dataset is not included in the github repo due to user privacy concerns, however it can be accessed from nlpg01 using the following command:
```
$ python3 src/test.py \
	--input_path /local1/baughan/dataset/political_tweets.csv \
	--output_path /local1/baughan/dataset/political_output.csv \
	--ER_model_path /local1/emazuh/output/reddit-emotion-pretrained.pth \
	--IP_model_path /local1/emazuh/output/reddit-interpretation-pretrained.pth \
	--EX_model_path /local1/emazuh/output/reddit-exploration-pretrained.pth
```

Once this has run, use `$ python3 src/analyze_political.py` to generate an output file named `political_outputc.csv`. The hand-annotated 50 examples referenced in the paper, their predicted values, and the F1 calculations can be found in `dataset/Political_Tweets_F1_calculations.csv` (in this repo).

Finally, the R Notebook in `src/political_correlation.Rmd` is used to assess levels of empathy in political ingroup and outgroup conversations. This relies on having [R](https://www.r-project.org/) and [RStudio](https://rstudio.com/products/rstudio/download/). To install any libraries in the beginning of the notebook, you can use `install.packages("<library name>")`.

#### Medical dataset
The medical dataset is not included in the github repo due to its large size. However, the data can be accessed and evaluated from nlpg01 using the following command:
```
$ python3 src/test.py \
	--input_path /local1/zsteineh/med_dataset/test_med_data.csv \
	--output_path /local1/zsteineh/med_dataset/med_data_output.csv \
	--ER_model_path /local1/emazuh/output/reddit-emotion-pretrained.pth \
	--IP_model_path /local1/emazuh/output/reddit-interpretation-pretrained.pth \
	--EX_model_path /local1/emazuh/output/reddit-exploration-pretrained.pth
```

After running the model on the data, you can use `$ python3 src/pull_hand_labeled.py` to get the statistics of the data and the model labels for the 39 hand-annotated samples. The hand-annotated samples and their F1 calculations can be found in `dataset/Med_dialogue_data/label_med_data.xls`

## Training Arguments

The training script accepts the following arguments: 

Argument | Type | Default value | Description
---------|------|---------------|------------
lr | `float` | `2e-5` | learning rate
lambda_EI | `float` | `0.5` | weight of empathy identification loss 
lambda_RE |  `float` | `0.5` | weight of rationale extraction loss
dropout |  `float` | `0.1` | dropout
data_percent | `float` | `1` | training data percent
max_len | `int` | `64` | maximum sequence length
batch_size | `int` | `32` | batch size
epochs | `int` | `4` | number of epochs
seed_val | `int` | `12` | seed value
train_path | `str` | `""` | path to input training data
dev_path | `str` | `""` | path to input validation data
test_path | `str` | `""` | path to input test data
do_validation | `boolean` | `False` | If set True, compute results on the validation data
do_test | `boolean` | `False` | If set True, compute results on the test data
no_domain_pretraining | `boolean` | `False` | If true, don't include domain adaptive pretraining
no_attention | `boolean` | `False` | If true, don't use attention head
save_model | `boolean` | `False` | If set True, save the trained model  
save_model_path | `str` | `""` | path to save model 


## Dataset Access Instructions

The Reddit portion of our collected dataset is available inside the [dataset](dataset) folder. The csv files with annotations on the three empathy communication mechanisms are `emotional-reactions-reddit.csv`, `interpretations-reddit.csv`, and `explorations-reddit.csv`. Each csv file contains six columns:
```
sp_id: Seeker post identifier
rp_id: Response post identifier
seeker_post: A support seeking post from an online user
response_post: A response/reply posted in response to the seeker_post
level: Empathy level of the response_post in the context of the seeker_post
rationales: Portions of the response_post that are supporting evidences or rationales for the identified empathy level. Multiple portions are delimited by '|'
```

We also test the model on the medical dataset from the paper below:
```bash
@article{chen2020meddiag,
  title={MedDialog: a large-scale medical dialogue dataset},
  author={Chen, Shu and Ju, Zeqian and Dong, Xiangyu and Fang, Hongchao and Wang, Sicheng and Yang, Yue and Zeng, Jiaqi and Zhang, Ruisi and Zhang, Ruoyu and Zhou, Meng and Zhu, Penghui and Xie, Pengtao},
  journal={arXiv preprint arXiv:2004.03329}, 
  year={2020}
}
```
The English part of the dataset, which we used, is available at: https://drive.google.com/drive/folders/1g29ssimdZ6JzTST6Y8g6h-ogUNReBtJD?usp=sharing

## Reproduced Results

The tables below show the reproducibility results of the main points of the paper.
![Reproducibility Results of Aims 1 and 2](https://github.com/emma-mens/Empathy-Mental-Health/blob/reproducibility/table_imgs/Aim1_reproduce.png?raw=True)
![Reproducibility Results of Aim 3](https://github.com/emma-mens/Empathy-Mental-Health/blob/reproducibility/table_imgs/Aim3_reproduce.png?raw=True)
