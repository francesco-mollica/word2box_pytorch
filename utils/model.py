from typing_extensions import Self
import torch.nn as nn
import torch
from box_embeddings.parameterizations.box_tensor import BoxFactory
from box_embeddings.modules import BoxEmbedding
from box_embeddings.initializations import UniformBoxInitializer
import pandas as pd
from math import dist
import matplotlib.patches as patches





global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1

class BoxModel(nn.Module):
    def __init__(self, emb_size, embedding_dim, box_vol, box_int, vocab, frequency_vocab):
        super(BoxModel, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        #self.initializer = UniformBoxInitializer(dimensions=embedding_dim, num_boxes=emb_size, box_type_factory=BoxFactory("sigmoid_from_vector"),
        #minimum=0, maximum=1, delta_max=1)
        self.embedding_dim = embedding_dim
        self.vocab_size = emb_size
        self.embeddings_word = BoxEmbedding(self.vocab_size, self.embedding_dim, box_factory=BoxFactory("sigmoid_from_vector"))
        self.embeddings_context = BoxEmbedding(self.vocab_size, self.embedding_dim, box_factory=BoxFactory("sigmoid_from_vector"))
        
        self.box_vol = box_vol
        self.box_int = box_int
        self.vocab = vocab
        self.frequency_vocab = frequency_vocab

    def forward(self, pos_u, pos_w, neg_w):

        embedding_u = self.embeddings_word(pos_u)
        embedding_w = self.embeddings_context(pos_w)
        embedding_negw = self.embeddings_context(neg_w)

        target_vol = self.box_vol(embedding_u)

        positive_vol = self.box_vol(embedding_w)

        negative_vol = self.box_vol(embedding_negw)

        positive_int_volumes = self.box_vol(self.box_int(embedding_w, embedding_u))

        negative_int_volumes = self.box_vol(self.box_int(embedding_negw, embedding_u))

        return target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes

    def extract_embeddings(self, boxes):

                list_box = []
                list_box.append(boxes.z.data.tolist())
                list_box.append(boxes.Z.data.tolist())
                rect = patches.Rectangle((list_box[0][0], list_box[0][1]), 
                            list_box[1][0] - list_box[0][0], 
                            list_box[1][1] - list_box[0][1])
                
                rx, ry = rect.get_xy()
                cx = rx + rect.get_width()/2.0
                cy = ry + rect.get_height()/2.0

                a = (0.5,0.5)
                b = (cx,cy)

                distance = dist(a,b)

                return rect, [list_box[0][0], list_box[0][1], list_box[1][0], list_box[1][1]], cx, cy, distance

    def most_similar(self, word, N=None):

        if N is None:
            N = len(self.vocab)

        embedding_all_target = self.model.embeddings_word.all_boxes

        try:
            index_word = (self.vocab[word])
            embedding_word = embedding_all_target[index_word]
            
            _, _, _, _, distance_word = self.extract_embeddings(embedding_word)

            volumes = self.box_vol(self.box_int(embedding_all_target, embedding_word))   

            idx = (-volumes).argsort()

            rows = []

            for i, index in enumerate(idx[0:N]):

                embedding_near = embedding_all_target[index]
                _, _, _, _, distance_near = self.extract_embeddings(embedding_near)

                rows.append([index_word, word, self.frequency_vocab[self.vocab.lookup_token(index_word)], distance_word,
                torch.exp(self.box_vol(embedding_word)).item(), self.vocab.lookup_token(index), self.frequency_vocab[self.vocab.lookup_token(index)], distance_near,  
                torch.exp(self.model.box_vol(embedding_near)).item() ])
        

            df = pd.DataFrame(rows, columns=["Ix", "Word", "Frequency", "Distance", "Volume", "Similar", "Frequency", "Distance", "Volume"])

            return df
        except:
            print("Word not in the dictionary") 
        


    def word_probability_similarity(self, word1, word2):

        embedding_all_target = self.model.embeddings_word.all_boxes

        try:
            index_word1 = (self.vocab[word1])
            index_word2 = (self.vocab[word2])

            word1 = embedding_all_target[index_word1]
            word2 = embedding_all_target[index_word2]

            score = self.box_vol(self.box_int(word1, word2))
            return score
        except:
            print("Words not in the dictionary")
