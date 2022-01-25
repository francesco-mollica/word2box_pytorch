import os
import yaml
import torch
import torch.optim as optim
from utils.model import BoxModel


def get_model_class(model_name: str):
    if model_name == "skipgram":
        return BoxModel
    else:
        raise ValueError("Choose model_name from: skipgram")
        return


def get_optimizer_class(name: str):
    if name == "Adam":
        return optim.Adam
    else:
        raise ValueError("Choose optimizer from: Adam")
        return
    

def save_config(config: dict, model_dir:str):
    """Save config file to `model_dir` directory"""
        
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)
        
        
def save_vocab(vocab, model_dir:str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
    