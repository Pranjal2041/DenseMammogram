# Deep Learning for Detection of Iso-Sense, Obscure Masses in Mammographically Dense Breasts
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/) 

## Introduction
Deep Learning for Detection of Iso-Sense, Obscure Masses in Mammographically Dense Breasts is a paper on object detection method for finding malignant masses in breast mammograms. Our model is particularly useful for dense breasts and iso-dense and obscure masses. In this paper we have included code and pretrained weights for the paper along with all the scripts to replicate numbers in the paper(Our private dataset is not included).  

## Getting Started


First clone the repo:
```bash
git clone https://github.com/Pranjal2041/DenseMammograms.git
```

Next setup the enviornment using `conda` or `virtualenv`: 
```bash 
1. conda create -n densebreast python=3.7
conda activate densebreast
pip install -r requirements.txt

or

2. python -m venv densebreast
source densebreast/bin/activate
pip install -r requirements.txt
```

## Pretrained Weights

You can download the pretrained models from this [url](https://csciitd-my.sharepoint.com/:f:/g/personal/cs5190443_iitd_ac_in/ElTbduIuI49EougSH05Tb4IBhbc5gXCrlok0X_xvAI196g?e=Ss2eS1) in the current directory.
<br>

## Running the Code

To generate predictions and FROC graphs using the pretrained models, run:
`python all_graphs.py`

For running individual models on other datasets, geenerate_{dataset}_preds.py have been provided.
For example to run predictions on inbreast, run:
`python geenerate_inbreast_preds.py`


## Demo

You can either use **Google Colab Demo** or **Huggingface demo**

## Citation

Details Coming Soon!

## License

TODO: Add License

