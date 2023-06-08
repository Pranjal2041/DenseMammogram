# Deep Learning for Detection of Iso-Sense, Obscure Masses in Mammographically Dense Breasts
<!-- [![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/) [![Open Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/)  -->

<div style='display:flex; gap: 0.25rem; '>
<a href='https://huggingface.co/spaces/Pranjal2041/DenseBreastCancerDetection'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> 
<a href='https://link.springer.com/article/10.1007/s00330-023-09717-7'><img src='https://img.shields.io/badge/Springer-Research%20Paper-red?style=plastic&logo=pubmed'></a> 
<a href='https://link.springer.com/article/10.1007/s00330-023-09717-7'><img src='https://img.shields.io/badge/Website%20-blue?style=plastic&logo=appveyor'></a> 
<!-- <img alt="AppVeyor" src="https://img.shields.io/appveyor/build/Pranjal2041/DenseMammogram"> -->
</div>



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


## Citation

```
TY  - JOUR
      AU  - Rangarajan, Krithika
      AU  - Agarwal, Pranjal
      AU  - Gupta, Dhruv Kumar
      AU  - Dhanakshirur, Rohan
      AU  - Baby, Akhil
      AU  - Pal, Chandan
      AU  - Gupta, Arun Kumar
      AU  - Hari, Smriti
      AU  - Banerjee, Subhashis
      AU  - Arora, Chetan
      PY  - 2023
      DA  - 2023/05/20
      TI  - Deep learning for detection of iso-dense, obscure masses in mammographically dense breasts
      JO  - European Radiology
      AB  - To analyze the performance of deep learning in isodense/obscure masses in dense breasts. To build and validate a deep learning (DL) model using core radiology principles and analyze its performance in isodense/obscure masses. To show performance on screening mammography as well as diagnostic mammography distribution.
      SN  - 1432-1084
      UR  - https://doi.org/10.1007/s00330-023-09717-7
      DO  - 10.1007/s00330-023-09717-7
      ID  - Rangarajan2023
      ER  - 
```

## License

The repo is Apache 2.0 licensed, as found in the [LICENSE](LICENSE) file.
