import argparse
import yaml
import os
import torch

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

from utils.constants import EMBED_DIMENSION, N_PRINT

def train(config):
    #os.makedirs(config["model_dir"])
    
    
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
    )

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    box_vol = Volume(volume_temperature=0.1, intersection_temperature=0.001)
    box_int = Intersection(intersection_temperature=0.001)

    model_class = get_model_class(config["model_name"])
    model = model_class(emb_size=vocab_size, embedding_dim=EMBED_DIMENSION, box_vol=box_vol, box_int=box_int)

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.most_similar(vocab, N_PRINT)
    trainer.train()
    print("Training finished.")
    trainer.most_similar(vocab, N_PRINT)

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)