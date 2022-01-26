# Word2Box in PyTorch

Implementation of one of the first word2box model using the box-embeddings library proposed by massach - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). 

## Word2box Overview

There 2 model architectures desctibed in the paper:


- Continuous Skip-gram Model (Skip-Gram), that predicts context for a word.

Difference with the original paper:

- Trained on [WikiText-2](https://pytorch.org/text/stable/datasets.html#wikitext-2) and [WikiText103](https://pytorch.org/text/stable/datasets.html#wikitext103) inxtead of Google News corpus.
- Context for both models is represented as 4 history and 4 future words.
- For CBOW model averaging for context word embeddings used instead of summation.
- For Skip-Gram model all context words are sampled with the same probability. 
- Plain Softmax was used instead of Hierarchical Softmax. No Huffman tree used either.
- Adam optimizer was used instead of Adagrad.
- Trained for 5 epochs.
- Regularization applied: embedding vector norms are restricted to 1.


### Skip-Gram Model in Details
#### High-Level Model
![alt text](docs/skipgram_overview.png)
#### Model Architecture
![alt text](docs/skipgram_detailed.png)


## Project Structure


```
.
├── README.md
├── config.yaml
├── notebooks
│   └── Inference.ipynb
├── requirements.txt
├── train.py
├── app.py
├── train.py
├── utils
│   ├── calculate_correlation.py
│   ├── dataloader.py
│   ├── helper.py
│   ├── inputdata.py
│   └── model.py
│   └── trainer.py  
│   └── word2vec_train.py
├── word_similarity_dataset
└── weights
```

- **utils/dataloader.py** - data loader for WikiText-2 and WikiText103 datasets
- **utils/model.py** - model architectures
- **utils/trainer.py** - class for model training and evaluation
- **utils/calculate_correlation.py** - script for calculate Spearman's correlation
- **utils/word2vec_train.py** - script for train a Gensim word2vec model
- **utils/helper.py** - contains some helper functions
- **utils/inputdata.py** - script that manipulate the data loader
- **app.py** - Dash app for visualize models
- **train.py** - script for training
- **config.yaml** - file with training parameters
- **weights/** - folder where expriments artifacts are stored
- **corpus/** - folder where txt corpus are saved
- **notebooks/toy_box_embeddings.ipynb** - demo of how box embeddings works and are used
- **word_similarity_dataset** - folder with all similarity dataset benchmarks

## Usage


```
python3 train.py --config config.yaml
```

Before running the command, change the training parameters in the config.yaml, most important:

- model_name ("skipgram")
- dataset ("WikiText2", "WikiText103")
- model_dir (directory to store experiment artifacts, should start with "weights/")


