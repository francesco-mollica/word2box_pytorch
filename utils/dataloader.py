from torchtext.data import to_map_style_dataset
from utils.inputdata import InputData
from torchtext.datasets import WikiText2, WikiText103
import string
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

def preprocessing(data_iter, RANGE):
    import re
    print("START PREPROCESSING")
    for i in range(0,RANGE):
        line = data_iter._data[i]
        line = re.sub(r'[^\w\s]', '', line)
        line = re.sub(r'[^\w]', ' ', line)
        line = line.replace('\n','')
        line = line.replace('\t','')
        line = line.replace('@-@','')
        line = line.replace('@','')
        line = line.strip()
        line = " ".join(line.split())
        line = line.lower()
        line = line.replace('unk', '')
        line = ' '.join([word for word in line.split() if word not in cachedStopWords])
        line.translate(str.maketrans('', '', string.punctuation))
        data_iter._data[i] = line

    print("FINISH PREPROCESSING")

def get_data_iterator(ds_name, ds_type, data_dir):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    
    data_iter = to_map_style_dataset(data_iter)
    #RANGE= len(data_iter)
    RANGE = 20000
    preprocessing(data_iter, RANGE)

    return data_iter[0:RANGE]

def get_dataloader_and_vocab(ds_name, ds_type, data_dir, batch_size, min_word_frequency, skipgram_n_words, neg_count):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    input_data = InputData(data_iter, batch_size, min_word_frequency, skipgram_n_words, neg_count)
    return input_data, input_data.final_vocab