import torch.nn as nn
import torch
from box_embeddings.parameterizations.box_tensor import BoxFactory
from box_embeddings.modules import BoxEmbedding

global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1

class BoxModel(nn.Module):
    def __init__(self, emb_size, embedding_dim, box_vol, box_int):
        super(BoxModel, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.embedding_dim = embedding_dim
        self.vocab_size = emb_size
        self.embeddings_word = BoxEmbedding(self.vocab_size, self.embedding_dim, box_factory=BoxFactory("sigmoid_from_vector"))
        self.embeddings_context = BoxEmbedding(self.vocab_size, self.embedding_dim, box_factory=BoxFactory("sigmoid_from_vector"))
        self.box_vol = box_vol
        self.box_int = box_int

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

    # DEFINE A WORD_SIMILARITY_PROBABILITY FUNCTION 

    # def word_probability_similarity(self, w1, w2):

    #     with torch.no_grad():
    #         word1 = self.embeddings_word(torch.LongTensor(w1).cuda())
    #         word2 = self.embeddings_word(torch.LongTensor(w2).cuda())

    #         score = self.box_vol(self.box_int(word1, word2))
    #         return score