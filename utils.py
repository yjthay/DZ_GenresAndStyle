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
import re
from sklearn.preprocessing import MultiLabelBinarizer

from datasets import load_dataset, list_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import multilabel_confusion_matrix, f1_score


# Load Reddit comments from list of list (from HuggingFace) run them through BERT Tokenzier and Model, using them to transform raw text input into PyTorch tensor
class EmotionsDataset(Dataset):
    def __init__(self, data, Model=BertModel, Tokenizer=BertTokenizer, max_length=12, bert_type='bert-base-cased',
                 device='cuda'):
        self.tokenizer = Tokenizer.from_pretrained(bert_type)
        self.model = Model.from_pretrained(bert_type).to(device)
        self.device = device

        # Loading the data into text and labels
        text, labels = data['text'], data['labels']

        # Tokenize the text and extract input_ids and attention_mask (ignoring
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        # Using sklearn MultiLabelBinarizer to change labels of [[3],[2],[0],[1,3]] to [[0,0,0,1],[0,0,1,0],[1,0,0,0],[0,1,0,1]]
        labels = label_multi_one_hot(labels).to(device)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, index):
        # full-BERT forward prop
        x = self.model(input_ids=self.input_ids[None, index], attention_mask=self.attention_mask[None, index])[0]
        x = x.view(np.prod(x.shape)).detach()  # detach full-BERT computation graph
        y = self.labels[index]
        return x.float(), y.float()

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
        sigmoid = torch.nn.Sigmoid()
        return sigmoid(outputs)


def best_model_filename(folder_path):
    regex_condition = re.compile('^.+_(.+\..+)\.pt')
    min_val_loss = np.inf
    model_name = ''
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            val_loss = re.findall(regex_condition, filename)
            if val_loss == []:
                continue
            val_loss = float(val_loss[0])
            if val_loss < min_val_loss:
                # if val_loss>min_val_loss:
                min_val_loss = val_loss
                model_name = filename
    return folder_path + model_name


def pickle_dump(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def label_multi_one_hot(list_of_list):
    mlb = MultiLabelBinarizer()
    index, columns = len(list_of_list), max(list_of_list)[0] + 1
    labels = pd.DataFrame(mlb.fit_transform(list_of_list), columns=range(columns), index=range(index))
    labels = torch.tensor(labels.values, dtype=float)
    return labels


def mapping(data=load_dataset('go_emotions')):
    label_mapping = {}
    obj = data['train'].features['labels'].feature
    num_classes = obj.num_classes
    for i in range(num_classes):
        label_mapping[i] = obj.int2str(i)
    return label_mapping


def predict(model, data, batch_size=32):
    '''
    outputs = torch.rand((3,3))
    threshold = 0.5
    y_pred = (outputs >= threshold)*torch.ones(outputs.shape)
    '''
    n = len(data)
    loader = DataLoader(data, batch_size=batch_size)
    pred = []
    for x, y in loader:
        y_pred = model(x)
        pred += y_pred.tolist()
    return np.array(pred)


def mlb_confusion_matrix(label_mapping, y_pred, y_true):
    '''
    :param label_mapping: dictionary of label_num to emotions mapping
    :param y_pred: numpy array of predict labels
    :param y_true: numpy array of true labels
    :return:
    dictionary of emotions and confusion matrix associated with it
    '''
    label_names = list(label_mapping.values())
    m_classes = len(label_names)
    output = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
    output = output / np.sum(output, axis=(1, 2))[:, None, None]
    return output


# from utils import *
#
# data = load_dataset('go_emotions')
# model_file_path = best_model_filename('model/epochs/64/')
# model = torch.load(model_file_path)
# label_mapping = mapping()
#
# test_data = EmotionsDataset(data['test'], max_length=64)
# test_confusion_matrix = mlb_confusion_matrix(label_mapping, model, test_data, batch_size=32)
# test_score = mlb_f1_score(model, test_data)
#
# train_data = EmotionsDataset(data['train'], max_length=64)
# train_confusion_matrix = mlb_confusion_matrix(label_mapping, model, train_data, batch_size=32)
# train_score = mlb_f1_score(model, train_data)
