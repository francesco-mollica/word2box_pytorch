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
        train_steps,
        checkpoint_frequency,
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
        self.train_steps = train_steps
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
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
        loss = - average_positive

        a = torch.unsqueeze(target_vol,1)
        b = a.expand(a.shape[0], self.neg_count)
        
        ########provare a togliere negative_vol come regolarizzazione######
        #average_negative = torch.sum(torch.squeeze(negative_int_volumes-b ,1) ,1)
        if self.neg_count == 1:
            average_negative = torch.squeeze(negative_int_volumes-(negative_vol+b) ,1)
        else:
            average_negative = torch.sum(torch.squeeze(negative_int_volumes-(negative_vol+b) ,1) ,1)
        loss +=  average_negative

        #print("loss: ",torch.sum(loss).item(), " pos: ", torch.sum(average_positive).item(), "neg: ", torch.sum(average_negative).item())
        return torch.sum(loss)

    def train(self):
        for epoch in range(0,1):
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

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)


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
        box_vol = Volume(volume_temperature=0, intersection_temperature=0)
        box_int = Intersection(intersection_temperature=0)
        
        lis_all = list(range(len(vocab)))

        words = ["brother", "friend", "mother", "father", "daughter", "son", "wife", "husband"]
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
                    table.append([ index, vocab.lookup_tokens([index])[0], "--", vocab.lookup_tokens([combination[1][0]])[0],  ])
                else:
                    table.append([ index, vocab.lookup_tokens([index])[0], vocab.lookup_tokens([combination[0][0]])[0], vocab.lookup_tokens([combination[1][0]])[0],  ])

            print(tabulate(table, headers=["INDEX", "WORD", "CONTAINED/OVERLAPPED", "NEAR"]))

            #idx = (-volumes).argsort()

            #for i, indexx in enumerate(idx[0:N]):
                #print(" si interseca con:", vocab.lookup_tokens([indexx]))

            
                
                        
