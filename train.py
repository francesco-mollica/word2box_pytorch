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
    get_lr_scheduler,
    save_config,
    save_vocab,
)
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(config):

    path = ('/epochs_' + str(config["epochs"]) + '_min_count_' + str(config["min_word_frequency"]) 
    + '_batch_size_' + str(config["train_batch_size"]) + '_embed_dim_' + str(config["embed_dimension"]) 
    +  '_lr_' + str(config["learning_rate"]) + '_window_' + str(config["skipgram_n_words"]) + '_neg_count_' + str(config["neg_count"]) )

    os.makedirs(config["model_dir"]+ path, exist_ok=True)
    
    train_dataloader, vocab = get_dataloader_and_vocab(
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

    box_vol = Volume(volume_temperature=0.1, intersection_temperature=0.01)
    box_int = Intersection(intersection_temperature=0.01)

    model_class = get_model_class(config["model_name"])
    model = model_class(emb_size=vocab_size, embedding_dim=config["embed_dimension"], box_vol=box_vol, box_int=box_int)

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader = train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir= (config["model_dir"] + path),
        model_name=config["model_name"],
        skipgram_n_words=config["skipgram_n_words"],
        neg_count=config["neg_count"],
    )

    trainer.most_similar(vocab, config["n_print"])
    trainer.train()
    print("Training finished.")
    trainer.most_similar(vocab, config["n_print"])

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, (config["model_dir"] + path))
    save_config(config, (config["model_dir"] + path))
    print("Model artifacts saved to folder:", (config["model_dir"]+ path))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    set_all_seeds(12345)
    train(config)