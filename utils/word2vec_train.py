import pandas as pd
import os 
import sys
import nltk
wpt = nltk.WordPunctTokenizer()
from gensim.models import Word2Vec
sys.path.append("../")



def train_save_word2vec(ds_name, emb_dim, lr, batch_size, epochs, skipgram_n_words, neg_count, min_count, direc):
    df = pd.read_csv("corpus/" + ds_name + ".txt", header=None)
    df = df.rename(columns={0: 'token'})
    tokenized = [wpt.tokenize(document) for document in df['token']]
    w2v_model = Word2Vec(vector_size=emb_dim*2, sg=1, min_count=min_count, window=skipgram_n_words, negative=neg_count, alpha=lr, min_alpha=lr, shrink_windows=False)
    w2v_model.build_vocab(tokenized)
    w2v_model.train(tokenized, total_examples=w2v_model.corpus_count, epochs=epochs)
    w2v_model.save((direc + "/word2vec.model"))
   