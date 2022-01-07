import os
import numpy as np
import json
import torch
from tqdm import tqdm
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection
from tabulate import tabulate
import itertools


class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
        skipgram_n_words,
        neg_count,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.neg_count = neg_count
        self.skipgram_n_words = skipgram_n_words
        self.loss = {"train": []}
        #self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def loss_(self, target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes):

        average_positive = ((positive_int_volumes)-(target_vol+positive_vol))
        #loss = torch.zeros(average_positive.shape[0]).cuda()

        a = torch.unsqueeze(target_vol,1)
        b = a.expand(a.shape[0], self.neg_count)
        
        ########provare a togliere negative_vol come regolarizzazione######
        #average_negative = torch.sum(torch.squeeze(negative_int_volumes-b ,1) ,1)
        if self.neg_count == 1:
            average_negative = torch.squeeze(negative_int_volumes-(negative_vol+b) ,1)
        else:
            average_negative = torch.sum(torch.squeeze(negative_int_volumes-(negative_vol+b) ,1) ,1)

        loss = ((-average_positive) +  average_negative)

        #print("loss: ",torch.sum(loss).item(), " pos: ", torch.sum(average_positive).item(), "neg: ", torch.sum(average_negative).item())
        return torch.sum(loss)


    def train(self):
        self.model.train()
        running_loss = []

        pair_count = self.train_dataloader.evaluate_pair_count(self.skipgram_n_words)
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
            #self.lr_scheduler.step()

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)


    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)

    def most_similar(self, vocab, N):
        box_vol = Volume(volume_temperature=0.1, intersection_temperature=0.01)
        box_int = Intersection(intersection_temperature=0.01)
        
        lis_all = list(range(len(vocab)))

        words = ["mother", "father"]
        lis = []
        for i in words:
            lis.append(vocab[i])
        
        
        pos = torch.tensor(lis_all).cuda()
        embedding_u = self.model.embeddings_word(pos)


        for i, index in enumerate(lis):
            pos = torch.tensor([index]).cuda()
            embedding = self.model.embeddings_word(pos)
            volumes = box_vol(box_int(embedding_u, embedding))

            near = []
            box_contenuto = []

            for i, val in enumerate(volumes.data):
                if (val.item() == box_vol(embedding).item() ) :
                    box_contenuto.append((i, box_vol(self.model.embeddings_word(torch.tensor([i]).cuda())).item()))
                else:
                    near.append((i, val.item()))

            table=[]
            ####di quelli in cui è contenuto o overlappato prendo 10 con volumi più piccoli#####
            box_contenuto.sort(key=lambda a: a[1], reverse=False)

            ####di quelli con cui è intersecato prendi quelli che hanno intersezione maggiore#####
            near.sort(key=lambda a: a[1], reverse=True)  

            for combination in itertools.zip_longest(box_contenuto[0:N], near[0:N]):
                if combination[0] == None:
                    table.append([ index, vocab.lookup_tokens([index])[0], box_vol(embedding),  
                    "  --", "  --", vocab.lookup_tokens([combination[1][0]])[0],  combination[1][1], box_vol(self.model.embeddings_word(torch.tensor([combination[1][0]]).cuda())).item()])
                else:

                    table.append([ index, vocab.lookup_tokens([index])[0], box_vol(embedding), 
                    vocab.lookup_tokens([combination[0][0]])[0], combination[0][1], vocab.lookup_tokens([combination[1][0]])[0],  
                    combination[1][1], box_vol(self.model.embeddings_word(torch.tensor([combination[1][0]]).cuda())).item() ])
            

            idx = (-volumes).argsort()
            for i, indexx in enumerate(idx[0:N]):
                print(" si interseca con:", vocab.lookup_tokens([indexx]))


            print(tabulate(table, headers=["INDEX", "WORD", "VOLUME", "CONTAINED/OVERLAPPED", "VOLUME",  "NEAR", "VOL_INTERSECTION", "VOLUME"]))

            

            

            
                
                        
