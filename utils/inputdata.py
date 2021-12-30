from typing import OrderedDict
import numpy as numpy
from collections import deque
import pickle
import torch
numpy.random.seed(12345)
from torch.utils.data import Dataset
from torchtext.vocab import vocab

class InputData(Dataset):
    """
    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, data_iter, batch_size, min_count, skipgram_n_words, neg_count):
        self.idx = 0
        self.data_iter = data_iter
        self.batch_size = batch_size
        self.skipgram_n_words = skipgram_n_words
        self.neg_count = neg_count
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
    
    def __len__(self):
        return (len(self.sentence_count))

    def get_words(self, min_count, load=False):
        self.sentence_length = 0
        self.sentence_count = 0
        self.word_count = 0
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        self.word_frequency_vocab = dict()
        self.final_vocab = None
        

        if load==False:
            word_frequency = dict()
            for line in self.data_iter:

                ######remove stopowords here######
                print("sentence_count:", self.sentence_count)
                self.sentence_count += 1
                line = line.strip().split(' ')
                if line==['']:
                    line=[]
                self.sentence_length += len(line)

                for w in line:
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

            infos = dict()
            infos["sentence_length"]=self.sentence_length
            infos["sentence_count"]=self.sentence_count
            infos["word_frequency"]=self.word_frequency
            infos["word_frequency_vocab"]=self.word_frequency_vocab
            infos["word_count"]=self.word_count
            infos["word2id"]=self.word2id
            infos["id2word"]=self.id2word

            with open('/home/fmollica/Downloads/infos.pickle', 'wb') as handle:
                pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("frequenze & infos salvate!")
        else:
            with open('/home/fmollica/Downloads/infos.pickle', 'rb') as handle:
                infos = pickle.load(handle)
                self.sentence_length=infos["sentence_length"]
                self.sentence_count=infos["sentence_count"]
                self.word_frequency=infos["word_frequency"]
                self.word_frequency_vocab=infos["word_frequency_vocab"]
                self.word_count=infos["word_count"]
                self.word2id=infos["word2id"]
                self.id2word=infos["id2word"]
                self.final_vocab = vocab(self.word_frequency_vocab)

            print("frequenze & infos caricate!")

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

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
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - self.skipgram_n_words, 0):i + self.skipgram_n_words + 1]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
            
            self.idx += 1
                
        batch_pairs = []
        for _ in range(self.batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())     
        neg_v = self.get_neg_v_neg_sampling(batch_pairs, self.neg_count)

        return torch.tensor(batch_pairs).T, torch.tensor(neg_v)


    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * self.skipgram_n_words - 1) - (
            self.sentence_count - 1) * (1 + self.skipgram_n_words) * self.skipgram_n_words