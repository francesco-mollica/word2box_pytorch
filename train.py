import argparse
import yaml
import torch
import random
import numpy as np
import os 
from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    save_config,
    save_vocab,
)
from utils.word2vec_train import train_save_word2vec
from utils.calculate_correlation import save_correlations_results
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection
from scipy.stats import reciprocal

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(config):
    
    list_batch_size_wikitext2 = [16,32,64,128,256]
    list_batch_size_wikitext103 = [2048,4096,8192,16384,32768]
    list_lr = np.exp(np.random.uniform(np.log(np.exp(-1)), np.log(np.exp(-10)), 2))
    print(list_lr)
    list_ws = [5,6,7,8,9,10]
    list_neg_samp = [2,5,10,20]

    while True:
        MYDIR = "corpus"
        CHECK_FOLDER = os.path.isdir(MYDIR)
        if not CHECK_FOLDER:
            os.makedirs(MYDIR, exist_ok=True)
            break
        
    while True:

        if config["dataset"]=="WikiText103":
            config["train_batch_size"] = random.choice(list_batch_size_wikitext103)
            config["min_word_frequency"] = 100
        else:
            config["train_batch_size"] = random.choice(list_batch_size_wikitext2)
            config["min_word_frequency"] = 50

        config["learning_rate"] = list_lr[0]
        config["skipgram_n_words"] = random.choice(list_ws)
        config["neg_count"] = random.choice(list_neg_samp)
        

        path = ('/epochs_' + str(config["epochs"]) + '_min_count_' + str(config["min_word_frequency"]) 
        + '_batch_size_' + str(config["train_batch_size"]) + '_embed_dim_' + str(config["embed_dimension"]) 
        +  '_lr_' + str(config["learning_rate"]) + '_window_' + str(config["skipgram_n_words"]) + '_neg_count_' + str(config["neg_count"]) )
        
        
        MYDIR = config["model_dir"]+ path
        CHECK_FOLDER = os.path.isdir(MYDIR)
        if not CHECK_FOLDER:
            os.makedirs(config["model_dir"]+ path, exist_ok=True)
            break
    
    train_dataloader, vocab, frequency_vocab = get_dataloader_and_vocab(
        ds_name=config["dataset"],
        ds_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        min_word_frequency=config["min_word_frequency"],
        skipgram_n_words=config["skipgram_n_words"],
        neg_count= config["neg_count"],
        save=True,
    )

    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")

    box_vol = Volume(volume_temperature=0, intersection_temperature=0)
    box_int = Intersection(intersection_temperature=0.001)

    model_class = get_model_class(config["model_name"])
    model = model_class(emb_size=vocab_size, embedding_dim=config["embed_dimension"], box_vol=box_vol, box_int=box_int, vocab=vocab, frequency_vocab=frequency_vocab)

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader = train_dataloader,
        optimizer=optimizer,
        device=device,
        model_dir= (config["model_dir"] + path),
        model_name=config["model_name"],
        skipgram_n_words=config["skipgram_n_words"],
        neg_count=config["neg_count"],
        emb_dim = config["embed_dimension"],
        lr = config["learning_rate"],
        min_count = config["min_word_frequency"],
    )
    
    trainer.save_model(type_="init")
    trainer.save_table(vocab=vocab, frequency_vocab = frequency_vocab, direc = config["model_dir"] + path, type_="init")
    print("INITIALIZATION DATA SAVED")
    trainer.train()
    if config["embed_dimension"] == 2:
        trainer.save_visuals(vocab = vocab, direc = config["model_dir"] + path, typ = "target")
        trainer.save_visuals(vocab = vocab, direc = config["model_dir"] + path, typ = "context")
    trainer.save_table(vocab=vocab, frequency_vocab = frequency_vocab, direc = config["model_dir"] + path, type_="final")
    print("TRAINING FINISHED AND VISUALIZATIONS SAVED.")
    print("START TRAINING WORD2VEC")
    train_save_word2vec(ds_name=config["dataset"], emb_dim = config["embed_dimension"], lr = config["learning_rate"],batch_size=config["train_batch_size"], 
    epochs= config["epochs"], skipgram_n_words=config["skipgram_n_words"], neg_count=config["neg_count"], 
    min_count = config["min_word_frequency"], direc = config["model_dir"] + path)
    print("FINISH TRAINING WORD2VEC")
    trainer.save_model("final")
    trainer.save_loss()
    save_vocab(vocab, (config["model_dir"] + path))
    save_config(config, (config["model_dir"] + path))
    print("SAVE CORRELATIONS RESULTS")
    save_correlations_results(direc = config["model_dir"] + path)
    print("MODEL ARTIFACTS SAVED TO: ", (config["model_dir"]+ path))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    #set_all_seeds(12345)
    train(config)