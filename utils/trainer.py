import os
import numpy as np
import json
import torch
from tqdm import tqdm
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection
import plotly.graph_objects as go
import pandas as pd


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

    def loss_(self, target_vol, positive_vol, negative_vol, positive_int_volumes, negative_int_volumes):

        average_positive = ((positive_int_volumes)-(target_vol+positive_vol))
        #average_positive = ((positive_int_volumes))
        #loss = torch.zeros(average_positive.shape[0]).cuda()

        a = torch.unsqueeze(target_vol,1)
        b = a.expand(a.shape[0], self.neg_count)
        
        if self.neg_count == 1:
            average_negative = torch.squeeze(negative_int_volumes-(negative_vol+b) ,1)
            #average_negative = torch.squeeze(negative_int_volumes ,1)
        else:
            average_negative = torch.sum(torch.squeeze(negative_int_volumes-(negative_vol+b) ,1) ,1)
            #average_negative = torch.sum(torch.squeeze(negative_int_volumes ,1) ,1)

        loss = ((-average_positive) +  average_negative)

        #print("loss: ",torch.sum(loss).item(), " pos: ", torch.sum(average_positive).item(), "neg: ", torch.sum(average_negative).item())
        return torch.mean(loss)


    def train(self):

        self.model.train()
        running_loss = []

        pair_count = self.train_dataloader.evaluate_pair_count(self.skipgram_n_words)
        batch_count = self.epochs * pair_count / self.train_dataloader.batch_size
        process_bar = tqdm(range(int(batch_count)))
        print("START TRAINING BOX-MODEL")
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

            embedding_target = boxes_target[i]
            _, _, _, _, distance_target = self.model.extract_embeddings(embedding_target)
            embedding_context = boxes_context[i]
            _, _, _, _, distance_context = self.model.extract_embeddings(embedding_context)



            rows.append([i, vocab.lookup_token(i), frequency_vocab[vocab.lookup_token(i)],
             torch.exp(self.model.box_vol(embedding_target)).item(), torch.exp(self.model.box_vol(embedding_context)).item(),
             distance_target, distance_context ])



        df = pd.DataFrame(rows, columns=["Ix", "Word", "Frequency", "Volume_Target", "Volume_Context", "Distance_Target", "Distance_Context"])

        df.to_pickle(direc + '/dataframe' + '_' + type_ +  '.pkl')
        #pivot_ui(df, outfile_path = direc + '/pivottablejs.html')
        #HTML('pivottablejs.html')


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
        #for i, point in enumerate(points):
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
        

        

            

            
                
                        
