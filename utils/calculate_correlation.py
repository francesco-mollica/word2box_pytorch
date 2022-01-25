import pandas as pd
import os 
import sys
import nltk
wpt = nltk.WordPunctTokenizer()
from gensim.models import Word2Vec
sys.path.append("../")
import torch
from scipy.stats import spearmanr
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume
import pickle

####calculate Spearman Rank correlation and corresponding p-value
def save_correlations_results(direc):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'{direc}/model_final.pt', map_location=device)
    #df = pd.read_pickle(f'./{folder}/{dataset}/dataframe_final.pkl')        
    #model = torch.load(f'{folder}/{dataset}/model_final.pt', map_location=device)
    vocab = torch.load(f'{direc}/vocab.pt')
    #word2vec_model = Word2Vec.load(f'{folder}/{dataset}/word2vec.model')
    word2vec_model = Word2Vec.load(f'{direc}/word2vec.model')

    volume = Volume(volume_temperature=0.0, intersection_temperature=0.0)
    intersection = Intersection(intersection_temperature=0.0)

    boxes = model.embeddings_word.all_boxes

    rows_data = []

    for filename in os.listdir("word_similarity_dataset"):
        dataframe = pd.read_csv("word_similarity_dataset/" + filename)
        dataframe.rename( columns={'Unnamed: 0':'index'}, inplace=True )

        list_similarity_word2box = []
        list_similarity_word2vec = []
        list_similarity_sota = []

        if filename!="men.csv":
            for i, row in dataframe.iterrows():
                try:
                    word1_word2vec = word2vec_model.wv.key_to_index[row["word1"]]
                    word2_word2vec = word2vec_model.wv.key_to_index[row["word2"]]
                    word1_word2box = vocab.lookup_indices([row["word1"]])[0]
                    word2_word2box = vocab.lookup_indices([row["word2"]])[0]
                except: 
                    continue
                

                list_similarity_word2vec.append(word2vec_model.wv.similarity(row["word1"], row["word2"]))
                list_similarity_sota.append(row["similarity"])
                list_similarity_word2box.append(torch.exp(volume(intersection(boxes[word1_word2box], boxes[word2_word2box]))).item())

            rho_vec, p_vec = spearmanr(list_similarity_word2vec, list_similarity_sota)
            rho_box, p_box = spearmanr(list_similarity_word2box, list_similarity_sota)



        else:
            for i, row in dataframe.iterrows():
                try:
                    word1_word2vec = word2vec_model.wv.key_to_index[row["word1"][:-2]]
                    word2_word2vec = word2vec_model.wv.key_to_index[row["word2"][:-2]]
                    word1_word2box = vocab.lookup_indices([row["word1"][:-2]])[0]
                    word2_word2box = vocab.lookup_indices([row["word2"][:-2]])[0]
                except: 
                    continue
                
                list_similarity_word2vec.append(word2vec_model.wv.similarity(row["word1"][:-2], row["word2"][:-2]))
                list_similarity_sota.append(row["similarity"])
                list_similarity_word2box.append(torch.exp(volume(intersection(boxes[word1_word2box], boxes[word2_word2box]))).item())

            rho_vec, p_vec = spearmanr(list_similarity_word2vec, list_similarity_sota)
            rho_box, p_box = spearmanr(list_similarity_word2box, list_similarity_sota)

        rows_data.append([filename[:-4], rho_vec, p_vec, rho_box, p_box])


    columns_ = ["dataset", "rho_vec", "p_vec", "rho_box", "p_box"]

    dataf = pd.DataFrame(rows_data, columns = columns_)

    dataf.to_pickle(direc + "/dataframe_correlations.pkl")

