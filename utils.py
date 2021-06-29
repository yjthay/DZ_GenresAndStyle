# !pip install transformers
# !pip install datasets
# !pip install osfclient
import csv
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import itertools
from sklearn.preprocessing import MultiLabelBinarizer

from datasets import load_dataset,list_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

def best_model_filename(folder_path):
    regex_condition = re.compile('^.+_(.+\..+)\.pt')
    # regex_condition = re.compile('^.+_(.+)_.+.pt')
    min_val_loss = np.inf
    # min_val_loss = 0
    model_name = ''
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            val_loss = re.findall(regex_condition, filename)
            if val_loss==[]:
                continue
            val_loss = float(val_loss[0])
            if val_loss<min_val_loss:
            # if val_loss>min_val_loss:
                min_val_loss=val_loss
                model_name = filename
    return model_save_path+model_name

def pickle_dump(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

# Load Reddit comments from list of list (from HuggingFace) run them through BERT Tokenzier and Model, using them to transform raw text input into PyTorch tensor
class EmotionsDataset(Dataset):
    def __init__(self, data_all, Model=BertModel, Tokenizer=BertTokenizer, max_length=12, bert_type='bert-base-cased', device='cuda'):
        self.tokenizer = Tokenizer.from_pretrained(bert_type)
        self.model = Model.from_pretrained(bert_type).to(device)
        self.device = device

        # Loading the data into text and labels
        text, labels = data_all['text'], data_all['labels']

        # Tokenize the text and extract input_ids and attention_mask (ignoring
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        # Using sklearn MultiLabelBinarizer to change labels of [[3],[2],[0],[1,3]] to [[0,0,0,1],[0,0,1,0],[1,0,0,0],[0,1,0,1]]
        labels = self.label_multi_one_hot(labels).to(device)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    @staticmethod
    def label_multi_one_hot(list_of_list):
        mlb = MultiLabelBinarizer()
        index, columns = len(list_of_list), max(list_of_list)[0]+1
        labels = pd.DataFrame(mlb.fit_transform(list_of_list), columns=range(columns), index=range(index))
        labels = torch.tensor(labels.values, dtype=float)
        return labels

    def __getitem__(self, index):
        # full-BERT forward prop
        x = self.model(input_ids=self.input_ids[None, index], attention_mask=self.attention_mask[None, index])[0]
        x = x.view(np.prod(x.shape)).detach()  # detach full-BERT computation graph
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)

# Fully connected layers after BERT
class Layers(torch.nn.Module):
    # A group of fully connected layers with adjustable layers size
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        # ReLU activations in hidden layers
        for i in range(len(self.dims) - 2):
            layer = self.layers[i]
            x = layer(x).clamp(min=0)
        # last layer no activation
        layer = self.layers[-1]
        outputs = layer(x)
        return outputs

