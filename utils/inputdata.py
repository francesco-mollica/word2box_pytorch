from platform import win32_edition
from typing import OrderedDict
import numpy as numpy
from collections import deque
import pickle
import torch
import random
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from tqdm import tqdm
class InputData(Dataset):
    """
    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, data_iter, batch_size, min_count, skipgram_n_words, neg_count, ds_name, RANGE):
        self.idx = 0
        self.ds_name = ds_name
        self.range = RANGE
        self.data_iter = data_iter
        self.batch_size = batch_size
        self.skipgram_n_words = skipgram_n_words
        self.neg_count = neg_count
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.count_all_pairs = 0
        #self.initTableDiscards()
        self.init_sample_table()
        print('Sentence Count: %d' % (self.sentence_count))
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
    
    def __len__(self):
        return (len(self.sentence_count))

    def get_words(self, min_count):
        self.sentence_length = 0
        self.sentence_count = 0
        self.word_count = 0
        self.total_word_count = 0
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        self.word_frequency_vocab = dict()
        self.final_vocab = None

        word_frequency = dict()
        for line in self.data_iter:
            self.sentence_count += 1
            line = line.strip().split(' ')
            if line==['']:
                line=[]
                self.sentence_length += 0
            else:
                self.sentence_length += len(line)

            for w in line:
                self.total_word_count += 1
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        
        wid = 0
        
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            self.word_frequency_vocab[w] = c
            wid += 1
        
        self.final_vocab = vocab(self.word_frequency_vocab)
        self.word_count = len(self.word2id)

            
    def initTableDiscards(self):
        t = 0.001
        f = numpy.array(list(self.word_frequency.values())) / self.total_word_count
        self.discards = 1 - numpy.sqrt(t / f)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)

        self.sample_table = numpy.array(self.sample_table)
        
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        
        neg_v_all = []
        # coppie = torch.tensor(pos_word_pair).T
        # coppie_0 = coppie[0,].tolist()
        # coppie_1 = coppie[1,].tolist()
        # list_0 = list(set(coppie_0))
        # list_1 = list(set(coppie_1))
        # list_total = list(set(list_0 + list_1))
        
        for elem in pos_word_pair:
            # coppie = []
            # for el in pos_word_pair:
            #     if elem[0] == el[0]:
            #         coppie.append(el)
            
            # coppie = torch.tensor(coppie).T
            # coppie_0 = coppie[0,].tolist()
            # coppie_1 = coppie[1,].tolist()
            # list_0 = list(set(coppie_0))
            # list_1 = list(set(coppie_1))
            # list_total = list(set(list_0 + list_1))
            list_total = [elem[0]]
            neg_v = numpy.random.choice(self.sample_table, size=(count)).tolist()
            not_contains = [target for target in neg_v if target not in list_total]
            contains = [target for target in neg_v if target in list_total]
            contains_1 = []
            not_contains_1 = not_contains
            
            if len(contains)==0:
                not_contains_1 = not_contains
            while len(not_contains_1)!=count:
                neg_v_1 = numpy.random.choice(self.sample_table, size=((count-len(not_contains_1)))).tolist()
                not_contains_1 = not_contains_1 + [target for target in neg_v_1 if target not in list_total]
        
            neg_v_all.append(not_contains_1)
        #neg_v = numpy.random.choice(self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v_all

    def __getitem__(self, idx):
        
        while len(self.word_pair_catch) < self.batch_size:
            if (self.idx == len(self.data_iter)):
                self.idx=0

            sentence = self.data_iter[self.idx]

            while(sentence == ''):
                self.idx += 1
                if (self.idx == len(self.data_iter)):
                    self.idx=0
                sentence = self.data_iter[self.idx]
                
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    # if numpy.random.rand() < (1 - self.discards[self.word2id[word]]):
                        word_ids.append(self.word2id[word])
                except:
                    continue
            
            line_tuple = []

            for i,elem in enumerate(word_ids):
                line_tuple.append((elem, i))

            for i, u in enumerate(line_tuple):
                skipgram_n_words = self.skipgram_n_words
                
                for j, v in enumerate(line_tuple[max(i - skipgram_n_words, 0):i + skipgram_n_words+1]):
                    assert u[0] < self.word_count
                    assert v[0] < self.word_count
                    if i==v[1]:
                        continue
                    self.word_pair_catch.append((u[0], v[0]))

            self.idx += 1
                
        batch_pairs = []
        for _ in range(self.batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())     
        neg_v = self.get_neg_v_neg_sampling(batch_pairs, self.neg_count)

        return torch.tensor(batch_pairs).T, torch.tensor(neg_v)

    def evaluate_pair_count(self, window_size):
        
        print("START COUNT OF PAIRS!!")
        
        for i in tqdm(range(0,self.range)):
            sentence = self.data_iter[i]
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    # if numpy.random.rand() < (1 - self.discards[self.word2id[word]]):
                        word_ids.append(self.word2id[word])
                except:
                    continue
            
            line_tuple = []

            for i,elem in enumerate(word_ids):
                line_tuple.append((elem, i))

            word_pair_catch_2 = []

            for i, u in enumerate(line_tuple):
                skipgram_n_words = self.skipgram_n_words
                
                for j, v in enumerate(line_tuple[max(i - skipgram_n_words, 0):i + skipgram_n_words+1]):
                    
                    if i==v[1]:
                        continue
                    word_pair_catch_2.append((u[0], v[0]))
                    
            self.count_all_pairs += len(word_pair_catch_2)
            
        print("TOTAL NUMBER OF PAIRS: ",self.count_all_pairs)
        print("FINISH COUNT OF PAIRS!!")
        return self.count_all_pairs
