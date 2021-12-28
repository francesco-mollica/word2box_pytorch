import os
import numpy as np
import json
import torch
from tqdm import tqdm

from utils.constants import N_PRINT, SKIPGRAM_N_WORDS, NEG_COUNT

class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        checkpoint_frequency,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.loss = {"train": []}
        #self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def loss_(self, target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes):

        average_positive = ((positive_int_volumes)-(target_vol+positive_vol))
        loss = - average_positive

        a = torch.unsqueeze(target_vol,1)
        b = a.expand(a.shape[0], NEG_COUNT)
        
        ########provare a togliere negative_vol come regolarizzazione######
        #average_negative = torch.sum(torch.squeeze(negative_int_volumes-b ,1) ,1)
        average_negative = torch.sum(torch.squeeze(negative_int_volumes-(negative_vol+b) ,1) ,1)
        loss +=  average_negative

        print("loss: ",torch.mean(loss).item(), " pos: ", torch.mean(average_positive).item(), "neg: ", torch.mean(average_negative).item())
        return torch.sum(loss)

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            #self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}".format(
                #"Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    #self.loss["val"][-1],
                )
            )

            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):

        self.model.train()
        running_loss = []

        pair_count = self.train_dataloader.evaluate_pair_count(SKIPGRAM_N_WORDS)
        batch_count = self.epochs * pair_count / self.train_dataloader.batch_size
        process_bar = tqdm(range(int(batch_count)))

        for i in process_bar:

            pos_pairs, neg_v = self.train_dataloader[i]
            pos_u = pos_pairs[0,]
            pos_v = pos_pairs[1,]


            ##inputs target
            inputs = pos_u.to(self.device)
            #context positive
            labels = pos_v.to(self.device)
            ###context negative
            negative = neg_v.to(self.device)

            self.optimizer.zero_grad()
            #####chiamo la forward del modello
            target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes = self.model(inputs, labels, negative)
            #####chiamo la mia loss
            loss = self.loss_(target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    # def _validate_epoch(self):
    #     self.model.eval()
    #     running_loss = []

    #     with torch.no_grad():
    #         for i, batch_data in enumerate(self.val_dataloader, 1):
    #             ##inputs target
    #             inputs = batch_data[0].to(self.device)
    #             #context positive
    #             labels = batch_data[1].to(self.device)
    #             ###context negative
    #             negative = batch_data[2].to(self.device)

    #             target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes = self.model(inputs, labels, negative)
    #             #####chiamo la mia loss
    #             loss = self.loss_(target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes)

    #             running_loss.append(loss.item())

    #             if i == self.val_steps:
    #                 break

    #     epoch_loss = np.mean(running_loss)
    #     self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)


    ########metodi per printare intersezione tra parole 
    def most_similar(self, vocab, N):
        
        lis_all = list(range(len(vocab)))

        print("brother, ", vocab["brother"])
        print("mother, ", vocab["mother"])
        print("father, ", vocab["father"])
        print("daughter, ", vocab["daughter"])
        print("son, ", vocab["son"])
        print("wife, ", vocab["wife"])
        print("friend, ", vocab["friend"])
        words = ["brother", "mother", "father", "daughter", "son", "wife", "friend"]
        lis = []
        for i in words:
            lis.append(vocab[i])
        
        
        pos = torch.tensor(lis_all).cuda()
        embedding_u = self.model.embeddings_word(pos)

        for i, index in enumerate(lis):
            pos = torch.tensor([index]).cuda()
            embedding = self.model.embeddings_word(pos)
            volumes = self.model.box_vol(self.model.box_int(embedding, embedding_u))
            idx = (-volumes).argsort()
            print("parola: ", index, ", ", vocab.lookup_tokens([index]))
            for i, indexx in enumerate(idx[0:N]):
                print(" si interseca con:", vocab.lookup_tokens([indexx]))
        
