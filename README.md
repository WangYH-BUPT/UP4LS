# UP4LS

- **Thanks to the editor and professional reviewers of *"Expert Systems with Applications"* for providing valuable comments on our paper!**

<img src=https://github.com/WangYH-BUPT/UP4LS/blob/master/Figs/1.jpg width=90% />

## 1. Conda Environment

- Python 3.6.9
- transformers 3.4.0
- sentencepiece 0.1.99
- torch 1.10.0
- tokenizers 0.9.2
- torchtext 0.6.0
- `pip install -r requirements.txt`

## 2. Directory of repository 

```
·
├── BERT               #(This paper uses BERT model for experiments.)
│   └── README.md
│
├── Baselines             #(HiTIN_ACL2023 and HypEmo_ACL2023)
│   ├── HiTIN_ACL2023
│   │   ├── config
│   │   ├── data
│   │   ├── data_modules
│   │   ├── helper
│   │   ├── models
│   │   ├── train_modules
│   │   ├── vocab
│   │   ├── README.md
│   │   ├── train.py
│   │   └── train_tem.py
│   └── HypEmo_ACL2023
│       ├── config
│       ├── data
│       ├── data_modules
│       ├── helper
│       ├── models
│       ├── train_modules
│       ├── vocab
│       ├── README.md
│       ├── train.py
│       └── train_tem.py
│
├── Benchmark  #(This directory contains the covers and stegos generated.)
│   ├── ArianaGrande
│   ├── BarackObama
│   ├── Britneyspears
│   ├── Cristiano
│   ├── Ddlovato
│   ├── Jimmyfallon
│   ├── Justinbieber
│   ├── KimKardashian
│   ├── Ladygaga
│   ├── Selenagomez
│   └── README.md
│
├── Code 
│   ├── data_bert.py
│   ├── data_prepare.py
│   ├── main_bert.py   # main (Run this)
│   └── network_bert.py
│
├── Figs               #(Including images from the repository description)
│   └── 1.jpg
│
├── README.md
├── random_select.py 
├── t-SNE.py
└── requirements.txt                 #(Necessary environment for the project)
```
