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
import matplotlib as mpl
import json


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


def visualize_scatter(data_2d, label_ids, label_mapping, figsize=(10, 10)):
    '''
    :param data_2d: tSNE Reduced Data np.array(n samples by 2)
    :param label_ids: multilabelled data list of one-hot encodings (n samples by labels)
    :param label_mapping: dictionary of id to emotions string
    :param figsize: tuple size of output graph
    :return: plt
    '''
    cm = plt.get_cmap('gist_rainbow')
    norm = mpl.colors.Normalize(vmin=0, vmax=max(label_mapping.keys()))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)

    plt.figure(figsize=figsize)
    plt.grid()

    # Data Cleaning
    x, y = [], []
    idx, labels = np.nonzero(label_ids)
    for i, j in zip(idx, labels):
        x.append(data_2d[i] + np.random.rand())
        y.append(j)
    data_2d, label_ids = np.array(x), np.array(y)

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=cmap.to_rgba(label_id + 1),
                    linewidth='1',
                    alpha=0.8,
                    label=label_mapping[label_id])
    plt.legend(loc='best')
    return plt


def convert_to_ekman(labels, ekman_fname='data/ekman_mapping.json', data=load_dataset('go_emotions')):
    # Retrieve the json of ekman emotion name to goemotions name mapping
    # Name to Name mapping
    with open(ekman_fname) as f:
        ekman_mapping = json.load(f)

    # Create data obj
    obj = data['train'].features['labels'].feature
    goemotions_to_ekman, idx = {obj.num_classes: len(ekman_mapping)}, 0

    # Create mapping of goemotions key to ekman key i.e. dictionary of 28 keys mapped down to 6 ekman emotions
    for idx in range(obj.num_classes):
        emotion = obj.int2str(idx)
        for id, emotions in enumerate(list(ekman_mapping.values())):
            if emotion in emotions:
                goemotions_to_ekman[idx] = id

    # Get list of lists from multi label data and map them down into list of lists for ekman
    output = []
    for multilabel in labels:
        temp = []
        for label in multilabel:
            temp.append(goemotions_to_ekman[label])
        # Append unique ekman index
        output.append(list(set(temp)))
    return output


def convert_to_sentiment(labels, sentiment_fname='data/sentiment_mapping.json', data=load_dataset('go_emotions')):
    # Retrieve the json of sentiment emotion name to goemotions name mapping
    # Name to Name mapping
    with open(sentiment_fname) as f:
        sentiment_mapping = json.load(f)

    # Create data obj
    obj = data['train'].features['labels'].feature
    goemotions_to_sentiment, idx = {obj.num_classes: len(ekman_mapping)}, 0

    # Create mapping of goemotions key to sentiment key i.e. dictionary of 28 keys mapped down to 3+1 sentiments
    for idx in range(obj.num_classes):
        emotion = obj.int2str(idx)
        for id, emotions in enumerate(list(sentiment_mapping.values())):
            if emotion in emotions:
                goemotions_to_sentiment[idx] = id

    # Get list of lists from multi label data and map them down into list of lists for ekman
    output = []
    for multilabel in labels:
        temp = []
        for label in multilabel:
            temp.append(goemotions_to_sentiment[label])
        # Append unique ekman index
        output.append(list(set(temp)))
    return output


def reverse_one_hot(labels):
    idx, labels = np.nonzero(labels)
    output = [[]]
    prev_i = 0
    for i, j in zip(idx, labels):
        if prev_i == i:
            output[-1].append(j)
        else:
            output.append([j])
        prev_i = i
    return output


def gen_tsne_values(high_dim_data):
    tsne_model = TSNE(perplexity=30,
                      n_components=2,
                      n_iter=1000,
                      random_state=23,
                      learning_rate=500,
                      init="pca")
    new_values = tsne_model.fit_transform(high_dim_data)
    return new_values
