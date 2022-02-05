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
from scipy import spatial
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

    volume = Volume(volume_temperature=0.1, intersection_temperature=0.0001)
    intersection = Intersection(intersection_temperature=0.0001)

    boxes = model.embeddings_word.all_boxes

    rows_data_int = []
    rows_data_w_int = []
    rows_data_centroids = []

    for filename in os.listdir("word_similarity_dataset"):
        dataframe = pd.read_csv("word_similarity_dataset/" + filename)
        dataframe.rename( columns={'Unnamed: 0':'index'}, inplace=True )

        list_similarity_word2box_intersection = []
        list_similarity_word2box_weighted_intersection = []
        list_similarity_word2box_centroids = []
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
                #####SIMPLE INTERSECTION BETWEEN BOXES#####
                list_similarity_word2box_intersection.append(torch.exp(volume(intersection(boxes[word1_word2box], boxes[word2_word2box]))).item())
                #####WEIGHTED INTERSECTION BETWEEN BOXES####
                list_similarity_word2box_weighted_intersection.append(torch.exp(volume(intersection(boxes[word1_word2box], boxes[word2_word2box]))-volume(boxes[word1_word2box])).item())
                #####COSINE SIMILARITY BETWEEN CENTROIDS#####
                centroid_word1 = ((boxes[word1_word2box].z + boxes[word1_word2box].Z)/2).tolist()
                centroid_word2 = ((boxes[word2_word2box].z + boxes[word2_word2box].Z)/2).tolist()
                list_similarity_word2box_centroids.append(1 - spatial.distance.cosine(centroid_word1, centroid_word2))
                

            rho_vec, p_vec = spearmanr(list_similarity_word2vec, list_similarity_sota)
            rho_box_int, p_box_int = spearmanr(list_similarity_word2box_intersection, list_similarity_sota)
            rho_box_w_int, p_box_w_int = spearmanr(list_similarity_word2box_weighted_intersection, list_similarity_sota)
            rho_box_centroids, p_box_centroids = spearmanr(list_similarity_word2box_centroids, list_similarity_sota)



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
                #####SIMPLE INTERSECTION BETWEEN BOXES#####
                list_similarity_word2box_intersection.append(torch.exp(volume(intersection(boxes[word1_word2box], boxes[word2_word2box]))).item())
                #####WEIGHTED INTERSECTION BETWEEN BOXES####
                list_similarity_word2box_weighted_intersection.append(torch.exp(volume(intersection(boxes[word1_word2box], boxes[word2_word2box]))-volume(boxes[word1_word2box])).item())
                #####COSINE SIMILARITY BETWEEN CENTROIDS#####
                centroid_word1 = ((boxes[word1_word2box].z + boxes[word1_word2box].Z)/2).tolist()
                centroid_word2 = ((boxes[word2_word2box].z + boxes[word2_word2box].Z)/2).tolist()
                list_similarity_word2box_centroids.append(1 - spatial.distance.cosine(centroid_word1, centroid_word2))

            rho_vec, p_vec = spearmanr(list_similarity_word2vec, list_similarity_sota)
            rho_box_int, p_box_int = spearmanr(list_similarity_word2box_intersection, list_similarity_sota)
            rho_box_w_int, p_box_w_int = spearmanr(list_similarity_word2box_weighted_intersection, list_similarity_sota)
            rho_box_centroids, p_box_centroids = spearmanr(list_similarity_word2box_centroids, list_similarity_sota)

        rows_data_int.append([filename[:-4], rho_vec, p_vec, rho_box_int, p_box_int])
        rows_data_w_int.append([filename[:-4], rho_vec, p_vec, rho_box_w_int, p_box_w_int])
        rows_data_centroids.append([filename[:-4], rho_vec, p_vec, rho_box_centroids, p_box_centroids])

    columns_ = ["dataset", "rho_vec", "p_vec", "rho_box", "p_box"]

    dataf_int = pd.DataFrame(rows_data_int, columns = columns_)
    dataf_w_int = pd.DataFrame(rows_data_w_int, columns = columns_)
    dataf_centroids = pd.DataFrame(rows_data_centroids, columns = columns_)

    dataf_int.to_pickle(direc + "/simple_intersection_correlations.pkl")
    dataf_w_int.to_pickle(direc + "/weighted_intersection_correlations.pkl")
    dataf_centroids.to_pickle(direc + "/centroids_correlations.pkl")

