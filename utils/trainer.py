import os
import numpy as np
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from math import dist
import random
from torch.autograd import Variable


class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        optimizer,
        device,
        model_dir,
        model_name,
        skipgram_n_words,
        neg_count,
        emb_dim, 
        lr,
        min_count,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.neg_count = neg_count
        self.skipgram_n_words = skipgram_n_words
        self.emb_dim = emb_dim
        self.lr = lr
        self.min_count = min_count
        self.loss = {"train": []}
        #self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    
    def loss_5(self, target_vol, positive_vol, positive_int_volumes, neg_count):

        average_positive = positive_int_volumes 
        loss = (-average_positive)
        return loss

    def loss_5_neg(self, target_vol, negative_vol, negative_int_volumes, neg_count):
       
        average_negative = negative_int_volumes 
        loss = (average_negative)
        return loss

    def train(self):

        self.model.train()
        running_loss = []
        pair_count = self.train_dataloader.evaluate_pair_count(self.skipgram_n_words)
        print(pair_count)
        batch_count = (self.epochs * pair_count) / self.train_dataloader.batch_size

        process_bar = tqdm(range(int(batch_count)))
        loss_best_positive = float('-inf')
        loss_positive = []

        #scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=(int(pair_count/self.train_dataloader.batch_size)*3), gamma=self.lr)

        print("START TRAINING BOX-MODEL")
        for i in process_bar:
            
            pos_pairs, neg_v = self.train_dataloader[i]
            pos_u = pos_pairs[0,]
            pos_v = pos_pairs[1,]


            ##inputs target
            inputs = Variable(pos_u).to(self.device)
            #context positive
            labels = Variable(pos_v).to(self.device)
            ###context negative
            negative = Variable(neg_v).to(self.device)


            self.optimizer.zero_grad()
            target_vol, positive_vol, positive_int_volumes = self.model.forward_pos(inputs, labels)
            tensor = torch.split(negative, 1,1)
            target_vol, negative_vol, negative_int_volumes = self.model.forward_neg(inputs, negative)
            loss_neg = self.loss_5_neg(target_vol, negative_vol, negative_int_volumes, self.neg_count).to(self.device)
            #target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes = model.forward(inputs, labels, negative)
            #with autograd.detect_anomaly():
            loss_pos = self.loss_5(target_vol, positive_vol, positive_int_volumes, self.neg_count).to(self.device)
            #loss = loss_5(target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes, 2)
            loss = (torch.mean(loss_pos) + torch.mean(torch.sum(loss_neg,1))).to(self.device)
            
            #if i%50 == 0:
                #print("LOSS POS : ", torch.mean((loss_pos)).item(), "LOSS NEG : ", torch.mean(torch.sum(loss_neg,1)).item())
            
            loss.backward()
            self.optimizer.step()
            #scheduler.step()

            loss_positive.append(torch.mean(loss_pos).item())
            running_loss.append(loss.item())
            if i%50==0:
                print(np.mean(loss_positive))
            if i%int(pair_count/self.train_dataloader.batch_size)==0:
                print(pos_pairs)
                
                model_path = os.path.join(self.model_dir, "checkpoint_" + 'epoch_' + str(int(i/int(pair_count/self.train_dataloader.batch_size))) + ".pt")

                if np.mean(loss_positive)>loss_best_positive:
                    dicty = {
                            'epoch': i,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss.item(),
                            'emb_size': self.model.vocab_size,
                            'emb_dim': self.model.embedding_dim,
                            'volume': self.model.box_vol,
                            'inter': self.model.box_int,
                            }
                    
                    best_model = self.model
                    loss_best_positive = np.mean(loss_positive)

                print(np.exp(np.mean(loss_best_positive)))
                epoch_loss = np.mean(running_loss)
                running_loss = []
                loss_positive = []
                self.loss["train"].append(epoch_loss)


        self.save_model(best_model, "final")    
        print("FINISH TRAINING BOX-MODEL")

    def save_model(self, type_):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model" + '_' + type_ + ".pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)


    def save_table(self, vocab, frequency_vocab, direc, type_):

        boxes_target = self.model.embeddings_word.all_boxes
        
        boxes_context = self.model.embeddings_context.all_boxes

        rows = []
        for i in range(len(vocab)):

            center = [0.5] * self.emb_dim

            embedding_target = boxes_target[i]
            centroid_target = ((embedding_target.z + embedding_target.Z)/2).tolist()
            distance_target = dist(center, centroid_target)

            embedding_context = boxes_context[i]
            centroid_context = ((embedding_context.z + embedding_context.Z)/2).tolist()
            distance_context = dist(center, centroid_context)
            
            rows.append([i, vocab.lookup_token(i), frequency_vocab[vocab.lookup_token(i)],
             torch.exp(self.model.box_vol(embedding_target)).item(), torch.exp(self.model.box_vol(embedding_context)).item(),
             distance_target, distance_context ])


        df = pd.DataFrame(rows, columns=["Ix", "Word", "Frequency", "Volume_Target", "Volume_Context", "Distance_Target", "Distance_Context"])

        df.to_pickle(direc + '/dataframe_infos_' + '_' + type_ +  '.pkl')
        

    def save_centroids(self, vocab, direc, typ, type_):
        if typ=="target":
            boxes = self.model.embeddings_word.all_boxes
        else:
            boxes = self.model.embeddings_context.all_boxes

        embeddings = []
        words = []

        #### append each box centroids 
        for i,elem in enumerate(boxes):
            box_centroid = ((elem.z + elem.Z)/2).tolist()            
            embeddings.append(box_centroid)
            words.append(vocab.lookup_tokens([i])[0])

        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.to_pickle(direc + '/dataframe' + '_' + typ +  '_centroids_' + type_ + '.pkl')
        # t-SNE transform
        tsne = TSNE(n_components=2)
        embeddings_df_trans = tsne.fit_transform(embeddings_df)
        embeddings_df_trans = pd.DataFrame(embeddings_df_trans)

        # get token order
        embeddings_df_trans.index = words

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=embeddings_df_trans[0],
                y=embeddings_df_trans[1],
                mode="text",
                text=embeddings_df_trans.index,
                textposition="middle center",
                textfont=dict(color="black")
            )
        )

        fig.update_layout(margin=dict(l=80, r=80, t=80, b=80), title={
                    'text' : ("epochs_" + str(self.epochs) + "_min_count_" + str(self.min_count) + "_batch_size_"  + str(self.train_dataloader.batch_size) 
                    + "_embed_dim_" +  str(self.emb_dim) + "_lr_" + str(self.lr) + "_window_" + str(self.skipgram_n_words) + "_neg_count_" + str(self.neg_count)),
                    'x':0.5,
                    'xanchor': 'center'
                } )

        fig.write_html(direc + '/word2box_' + typ + '_centroids_' + type_ + '_visualization.html')

    def save_visuals(self, vocab, direc, typ):
        if typ=="target":
            boxes = self.model.embeddings_word.all_boxes
        else:
            boxes = self.model.embeddings_context.all_boxes


        lis_all = list(range(len(vocab)))
        center_x = []
        center_y = []
        words = []
        all_rect = []
        points = []

        for i in lis_all:
            emb = boxes[i]
            rect, p, cx, cy, _ = self.model.extract_embeddings(emb)
            all_rect.append(rect)
            points.append(p)
            center_x.append(cx)
            center_y.append(cy)
            words.append(vocab.lookup_tokens([i])[0])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=center_x,
                y=center_y,
                mode="text",
                text=words,
                textposition="middle center",
                textfont=dict(color="black"),
            )
        )

        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])


        ###### DISEGNA I BOX...IMPIEGA TANTO TEMPO ########
        # for i, point in enumerate(points):
        #  color = "#%06x" % random.randint(0, 0xFFFFFF)
        #  fig.add_shape(type="rect",
        #      x0=point[0], y0=point[1], x1=point[2], y1=point[3],
        #      line=dict(color=color),
        #  )

        fig.update_layout(margin=dict(l=80, r=80, t=80, b=80), title={
                    'text' : ("epochs_" + str(self.epochs) + "_min_count_" + str(self.min_count) + "_batch_size_"  + str(self.train_dataloader.batch_size) 
                    + "_embed_dim_" +  str(self.emb_dim) + "_lr_" + str(self.lr) + "_window_" + str(self.skipgram_n_words) + "_neg_count_" + str(self.neg_count)),
                    'x':0.5,
                    'xanchor': 'center'
                } )

        fig.write_html(direc + '/word2box_' + typ + '_visualization.html')
        

        

            

            
                
                        
