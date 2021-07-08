#!/usr/bin/env python
# coding: utf-8
from typing import List, Optional, Union
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl
import logging
from pytorch_lightning.metrics.functional import f1 as f1_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

gtind2word ={}
gind2word = {}

#https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html

class dataset(Dataset):
    def __init__(self,trainlist,targetlist):
        self.trainlist = trainlist
        self.targetlist = targetlist
    def __len__(self):
        return len(self.trainlist)
    def __getitem__(self,idx):
        return self.trainlist[idx],self.targetlist[idx]

class NERDataModule(pl.LightningDataModule):
    def __init__(self,input_seq,target_seq,glove_emb_vec,batch_size = 32):
        super().__init__()       
        
        
        self.batch_size = batch_size
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.glove_emb_vec = glove_emb_vec

    def setup(self,stage: Optional[str] = None):
        self.train_seqdataset = dataset(self.input_seq[:1000],self.target_seq[:1000])
        self.valid_seqdataset = dataset(self.input_seq[1000:],self.target_seq[1000:])

    def pad_collate(self,batch):
        (xx, yy) = zip(*batch)    
        xx = list(xx)  
        yy = list(yy)  
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]  
        xx_tensor = [torch.LongTensor(x) for x in xx]
        yy_tensor = [torch.LongTensor(y) for y in yy]
        xx_pad = pad_sequence(xx_tensor, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy_tensor, batch_first=True, padding_value=0)
        xx_embed = self.glove_emb_vec[xx_pad]
        return xx_embed,xx_pad, yy_pad, x_lens, y_lens

    def train_dataloader(self):         
        
        train_data_loader = DataLoader(dataset=self.train_seqdataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.pad_collate)
        return train_data_loader

    def val_dataloader(self):                
        
        valid_data_loader = DataLoader(dataset=self.valid_seqdataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.pad_collate)
        return valid_data_loader


class DataPrep():
    def __init__(self,data_path="C:\\Users\\Ant PC\\Music\\dataset\\FIN5.txt",
    glove_path="C:\\Users\\Ant PC\\Music\\NER-LSTM-CNN-Pytorch-master\\NER-LSTM-CNN-Pytorch-master\\data\\glove.6B.100d.txt",
    embed_dim=100):       
        
        self.data_path = data_path
        self.glove_path = glove_path       
        self.embed_dim = embed_dim             
    
    def download_data(self):
        word_list = []
        target_list=  []
        word_str = ""
        target_str = ""
        with open(self.data_path,'r', encoding="utf8") as myfile:    
            for i, line in enumerate(myfile):        
                if ((len(line)>0 and i>0) and line not in ['\n', '\r\n']) :            
                    if (len(line)>0):
                        word_str = word_str+" "+line.split()[0]
                        target_str = target_str+" "+line.split()[-1]                    
                if line in ['\n', '\r\n']:            
                    word_list.append(word_str[1:])
                    target_list.append(target_str[1:])
                    word_str=""
                    target_str=""
        for wl,tl in zip(word_list[1:],target_list[1:]):
            assert len(wl.split()) == len(wl.split())
        return word_list,target_list


    def load_glove(self,emb_dim = 100):
        file = open(self.glove_path,'r',encoding='utf-8')    
        glove_emb = {} 
        word2ind ={}
        ind2word = {}
        # Include padding to the glove vocab
        glove_emb["<pad>"] = np.zeros(emb_dim,)
        emb_vec = [[0]*emb_dim]
        word2ind["<pad>"] = 0
        ind2word[0] = "<pad>"
        
        for i,text in enumerate(file):    
            i = i+1
            line = text.split()
            word = line[0]
            emb = np.array([float(val) for val in line[1:]])
            embl = [float(val) for val in line[1:]]        
            glove_emb[word] = emb     
            tmp_emb_vec = emb.reshape(1,-1)
            emb_vec.append(embl)
            word2ind[word] = i
            ind2word[i] = word
            
        #emb_vec = np.array(emb_vec)
        return glove_emb,emb_vec,word2ind,ind2word     

    def get_emb_vect(self,text,glove_emb_dict,glove_emb_list,word2ind,ind2word):
        words_NOT_found = 0
        embedding_dim = self.embed_dim
        for sent in text:
            for word in sent.split():        
                word = word.lower()
                try: 
                    glove_emb_dict[word]            
                except KeyError:
                    words_NOT_found += 1            
                    glove_emb_dict[word] = np.random.normal(scale=0.6, size=(embedding_dim,)) # scale is the S.D of the Gaussian Distribution(-1,1)
                    glove_emb_list.append(list(glove_emb_dict[word]))
                    word2ind[word] = len(glove_emb_dict)-1
                    ind2word[len(glove_emb_dict)-1] = word
        glove_emb_vec = np.array(glove_emb_list)
        return glove_emb_vec,glove_emb_dict,word2ind,ind2word
    
    def prepare_data(self):
        initglove_emb_dict,initglove_emb_list,initword2ind,initind2word = self.load_glove()
        word_list,target_list = self.download_data()
        text = word_list[1:] # First word is blank
        glove_emb_vec,glove_emb_dict,word2ind,ind2word = self.get_emb_vect(text,initglove_emb_dict,initglove_emb_list,initword2ind,initind2word)        
        target = target_list[1:]
        tind2word ={}
        tword2ind = {}
        uniq_words = ""
        maxlen = 0
        for sent in target:    
            uniq_words = uniq_words+ " " +sent    
            if len(sent.split()) > maxlen:
                maxlen = len(sent.split())        
        uniq_words = set(uniq_words.strip().split())
        tind2word[0] = "<pad>"
        tword2ind["<pad>"] = 0
        for i,word in enumerate(uniq_words):    
            i = i+1
            tind2word[i] = word.lower()
            tword2ind[word.lower()] = i
        input_seq = text
        target_seq = target
        """Now we can convert our input and target sequences to sequences of integers instead of characters by
        mapping them using the dictionaries we created above. This will allow us to one-hot-encode our input sequence subsequently."""
        for i in range(len(text)):
            input_seq[i] = [word2ind[word.lower()] for word in input_seq[i].split()]
            target_seq[i] = [tword2ind[word.lower()] for word in target_seq[i].split()]        

        return input_seq,target_seq,glove_emb_vec,ind2word,tind2word
    
class Model(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,batch_size,lr):
        super(Model, self).__init__()

        # Defining some parameters
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.num_class = output_size
        self.lr = lr

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True,bidirectional=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*2, output_size)
    
    def forward(self, x,x_lens):
        
        batch_size = self.batch_size
        #x_embed = self.embedding(x)
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        #Initializing hidden state for first input using method defined below
        hidden,cin = self.init_hidden(batch_size)
        #x = x.detach()
        # Passing in the input and hidden state into the model and obtaining outputs
        #out, (hidden,cout) = self.rnn(x, (hidden,cin))        
        output_packed, (hidden,cout) = self.rnn(x_packed, (hidden,cin))
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = output_padded.contiguous().view(-1, self.hidden_dim*2)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim,dtype=torch.float, device=self.device)
        cin = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim,dtype=torch.float,device=self.device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden,cin

    def training_step(self,batch: dict, batch_idx: int):
        loss,f1 = self.calc_loss(batch, batch_idx)
        self.log("train_loss",loss,on_step=True)
        self.log("train_f1",f1,on_step=True)
        return loss
    
    def validation_step(self,batch: dict, batch_idx: int):
        loss,f1 = self.calc_loss(batch, batch_idx)
        self.log("val_loss",loss,on_epoch=True,prog_bar=True)
        self.log("val_f1",f1,on_epoch=True,prog_bar=True)
        return loss
    
    def calc_loss(self, batch: dict, batch_idx: int):
        xx_pad,x_padded, y_padded, x_lens, y_lens = batch
        xx_pad = torch.tensor(xx_pad,dtype=torch.float,device=self.device)                
        output, hidden = self(xx_pad,x_lens)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(output, y_padded.view(-1).long())
        prob = nn.functional.softmax(output, dim=1).data
        # Taking the class with the highest probability score from the output    
        _,word_ind = torch.max(prob, dim=1) 
        mask = y_padded.reshape(-1) != 0
        pred_no_pad, labels_no_pad = word_ind[mask], y_padded.reshape(-1)[mask]
        f1 = f1_score(pred_no_pad, labels_no_pad, num_classes=self.num_class, average="macro")
        return loss, f1   


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.     

    data_prep = DataPrep(cfg.data.data_path,cfg.data.glove_path,cfg.model.embedding_dim)
    input_seq,target_seq,glove_emb_vec,ind2word,tind2word = data_prep.prepare_data()    
    output_tag = len(tind2word)
    vocab_size = len(ind2word)        
    pl.seed_everything(0)
    wandb_logger = pl.loggers.WandbLogger()
    # Instantiate the model with hyperparameters
    

    model = Model(input_size=cfg.model.embedding_dim, output_size=output_tag, hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,batch_size=cfg.model.batch_size,lr=cfg.model.lr)
    trainer = pl.Trainer(**cfg.trainer,logger=wandb_logger)
    # Simple logging of the configuration    
    trainer.fit(model,datamodule=NERDataModule(input_seq,target_seq,glove_emb_vec,batch_size=cfg.model.batch_size))       
    logger.info(OmegaConf.to_yaml(cfg)) 

    


"""def predict(model, text,do_print=False):
    or_text = text.copy()
    glove_emb_vec,glove_emb_dict,word2ind,ind2word = get_emb_vect(text,initglove_emb_dict,initglove_emb_list,initword2ind,initind2word)
    input_seq = text
    
    for i in range(len(text)):
        input_seq[i] = [word2ind[word.lower()] for word in text[i].split()]
    
    xx_tensor = [torch.LongTensor(x) for x in input_seq]    
    x_padded = pad_sequence(xx_tensor, batch_first=True, padding_value=0)    
    x_lens = [len(x) for x in text]    
    input_text = glove_emb_vec[x_padded]
    input_text = torch.tensor(input_text,dtype=torch.float).to(device)
    model.eval() # eval mode    
    out, hidden = model(input_text,x_lens)    
    prob = nn.functional.softmax(out, dim=1).data
    # Taking the class with the highest probability score from the output    
    _,word_ind = torch.max(prob, dim=1)    
    #print(word_ind)
    final_ind = []
    for line in range(len(or_text)):                
        sent = or_text[line]            
        for i in range(len(sent.split())):            
            if do_print:
                print(f"{sent.split()[i]} : {tind2word[word_ind[i].item()]}")
            final_ind.append(word_ind[i].cpu().numpy().item())
            
    return final_ind,hidden"""

if __name__ == "__main__":
    train()




