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
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
)
from sklearn.metrics import multilabel_confusion_matrix, f1_score
import matplotlib as mpl
import json
from sklearn.manifold import TSNE


# Load Reddit comments from list of list (from HuggingFace) run them through BERT Tokenzier and Model, using them to transform raw text input into PyTorch tensor
class EmotionsDataset(Dataset):
    def __init__(self, data, type=None, max_length=64, bert_type='bert-base-cased', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.device = device

        # Loading the data into text and labels
        if type == 'ekman':
            text, labels = data['text'], convert_to_ekman(data['labels'])
        elif type == 'senti':
            text, labels = data['text'], convert_to_sentiment(data['labels'])
        else:
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
        input_ids, attention_mask = self.input_ids[index], self.attention_mask[index]
        y = self.labels[index]
        return input_ids, attention_mask, y.float()

    def __len__(self):
        return len(self.labels)


# Fully connected layers after BERT
class BERT_Model(torch.nn.Module):
    # A group of fully connected layers with adjustable layers size
    def __init__(self, dims, bert_type='bert-base-cased', device='cuda'):
        super().__init__()
        self.dims = dims
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1], device=device) for i in range(len(dims) - 1)]
        )
        self.model = AutoModel.from_pretrained(bert_type).to(device)

    def forward(self, input_ids, attention_mask):
        # ReLU activations in hidden layers
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        batch, max_length, hidden = x.shape
        x = x.view(batch, max_length * hidden).float()
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
    # Create go-emotions label mapping keys
    label_mapping = {}
    obj = data['train'].features['labels'].feature
    num_classes = obj.num_classes
    for i in range(num_classes):
        label_mapping[i] = obj.int2str(i)
    return label_mapping


# Create ekman label mapping keys
def json_mapping(fname='data/ekman_mapping.json'):
    # fname='data/sentiment_mapping.json'
    with open(fname) as f:
        mapping = json.load(f)
    num_to_str = {}
    for idx, key in enumerate(mapping):
        num_to_str[idx] = key
    return num_to_str


def str_to_num(labels_dict):
    rev_labels_dict = dict((y, x) for x, y in labels_dict.items())
    return rev_labels_dict


def predict(model, data, batch_size=32):
    '''
    outputs = torch.rand((3,3))
    threshold = 0.5
    y_pred = (outputs >= threshold)*torch.ones(outputs.shape)
    '''
    n = len(data)
    loader = DataLoader(data, batch_size=batch_size)
    pred = []
    model.eval()
    for input_ids, attention_mask, y in loader:
        with torch.no_grad():
            y_pred = model(input_ids, attention_mask)
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
    norm = mpl.colors.Normalize(vmin=0, vmax=max(label_mapping.keys()) + 1)
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
    goemotions_to_ekman, idx = {obj.num_classes - 1: len(ekman_mapping)}, 0

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
    goemotions_to_sentiment, idx = {obj.num_classes - 1: len(sentiment_mapping)}, 0

    # Create mapping of goemotions key to sentiment key i.e. dictionary of 28 keys mapped down to 3+1 sentiments
    for idx in range(obj.num_classes):
        emotion = obj.int2str(idx)
        for id, emotions in enumerate(list(sentiment_mapping.values())):
            if emotion in emotions:
                goemotions_to_sentiment[idx] = id

    # Get list of lists from multi label data and map them down into list of lists for sentiment
    output = []
    for multilabel in labels:
        temp = []
        for label in multilabel:
            temp.append(goemotions_to_sentiment[label])
        # Append unique sentiment index
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


class T5Dataset(Dataset):
    def __init__(self, data, goemo_ratio=1 / 3, type="all", device='cuda'):
        super(T5Dataset, self).__init__()
        self.device = device
        self.texts, self.labels = data['text'], data['labels']

        self.tokenizer = config.TOKENIZER
        self.txt_max_length = config.TXT_MAX_LENGTH
        self.tgt_max_length = config.TGT_MAX_LENGTH
        self.goemo_mapping, self.ekman_mapping, self.senti_mapping = config.GOEMO_MAPPING, config.EKMAN_MAPPING, config.SENTI_MAPPING

        self.x, self.y = [], []
        for text, label in zip(self.texts, self.labels):
            choice = np.random.rand()
            if choice <= goemo_ratio:
                self.x.append(self._process_text("goemo", [text]))
                self.y.append(self._process_text("goemo", self._numeric_to_t5([label], self.goemo_mapping)))
            elif goemo_ratio < choice < goemo_ratio + (1 - goemo_ratio / 2.):
                self.x.append(self._process_text("ekman", [text]))
                self.y.append(
                    self._process_text("ekman", self._numeric_to_t5(convert_to_ekman([label]), self.ekman_mapping)))
            else:
                self.x.append(self._process_text("senti", [text]))
                self.y.append(
                    self._process_text("senti", self._numeric_to_t5(convert_to_sentiment([label]), self.senti_mapping)))

        self.texts, self.labels = list(itertools.chain(*self.x)), list(itertools.chain(*self.y))
        # self.goemo_texts, self.goemo_labels = self._process_text("goemo", self.texts), self._process_text("goemo",
        #                                                                                                   self._numeric_to_t5(
        #                                                                                                       self.labels,
        #                                                                                                       self.goemo_mapping))
        # self.ekman_texts, self.ekman_labels = self._process_text("ekman", self.texts), self._process_text("ekman",
        #                                                                                                   self._numeric_to_t5(
        #                                                                                                       self.ekman_labels,
        #                                                                                                       self.ekman_mapping))
        # self.senti_texts, self.senti_labels = self._process_text("senti", self.texts), self._process_text("senti",
        #                                                                                                   self._numeric_to_t5(
        #                                                                                                       self.senti_labels,
        #                                                                                                       self.senti_mapping))
        # if type.lower() == 'goemo':
        #     self.texts = list(itertools.chain(self.goemo_texts))
        #     self.labels = list(itertools.chain(self.goemo_labels))
        # elif type.lower() == 'ekman':
        #     self.texts = list(itertools.chain(self.ekman_texts))
        #     self.labels = list(itertools.chain(self.ekman_labels))
        # elif type.lower() == 'senti':
        #     self.texts = list(itertools.chain(self.senti_texts))
        #     self.labels = list(itertools.chain(self.senti_labels))
        # else:
        #     self.texts = list(itertools.chain(self.goemo_texts, self.ekman_texts, self.senti_texts))
        #     self.labels = list(itertools.chain(self.goemo_labels, self.ekman_labels, self.senti_labels))

    @staticmethod
    def _numeric_to_t5(labels, label_dict):
        # labels input as list of lists
        output = []
        for sample in labels:
            temp = []
            for label in sample:
                temp.append(label_dict[label])
            # output as "anger, annoyance" from ["anger" , "annoyance"]
            output.append(' , '.join(temp))
        return output

    @staticmethod
    def _process_text(pre_text, list_of_text):
        output = []
        for text in list_of_text:
            output.append(pre_text + ": " + text)
        return output

    def __getitem__(self, index):
        txt_tokenized = self.tokenizer.encode_plus(self.texts[index],
                                                   max_length=self.txt_max_length,
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_attention_mask=True,
                                                   return_token_type_ids=False,
                                                   return_tensors='pt')
        txt_input_ids = txt_tokenized['input_ids'].to(self.device)
        txt_attention_mask = txt_tokenized['attention_mask'].to(self.device)

        tgt_tokenized = self.tokenizer.encode_plus(self.labels[index],
                                                   max_length=self.tgt_max_length,
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_attention_mask=True,
                                                   return_token_type_ids=False,
                                                   return_tensors='pt')
        tgt_input_ids = tgt_tokenized['input_ids'].to(self.device)
        tgt_attention_mask = tgt_tokenized['attention_mask'].to(self.device)

        return (txt_input_ids.squeeze(),
                txt_attention_mask.squeeze(),
                tgt_input_ids.squeeze(),
                tgt_attention_mask.squeeze(),
                self.texts[index],
                self.labels[index])

    def __len__(self):
        return len(self.labels)


class T5Model(torch.nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()

        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                lm_labels=None):
        return self.t5_model(input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             labels=lm_labels)


# Function for training
def train_T5(model, data, ratio, epochs, lr, batch_size, show_progress=False, save_path=None):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Construct data loader from training and validation dataset
    val_loader = DataLoader(T5Dataset(data['validation'], goemo_ratio=1.0), batch_size=batch_size)

    val_losses = []
    train_losses = []

    best_model, best_val_loss = None, np.inf
    # Training
    for epoch in range(epochs):
        # backprop
        train_loader = DataLoader(T5Dataset(data['train'], goemo_ratio=ratio), batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        inner_iter = 0
        pbar = tqdm(train_loader, position=0, leave=True)
        epoch_idx = int(epoch + 1)
        model.train()
        for x_inputs, x_masks, y_inputs, y_masks, _, _ in pbar:
            pbar.set_description("Processing Epoch %d" % epoch_idx)

            lm_labels = y_inputs
            lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

            optimizer.zero_grad()
            outputs = model(input_ids=x_inputs,
                            attention_mask=x_masks,
                            lm_labels=lm_labels,
                            decoder_attention_mask=y_masks)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # calculate training loss
        train_loss = running_loss / len(train_loader)  # calculate validation loss

        # y_pred, y_true = [], []
        with torch.no_grad():
            running_loss = 0.0
            for x_val_inputs, x_val_masks, y_val_inputs, y_val_masks, x_text, y_labels in val_loader:
                lm_labels = y_val_inputs
                lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100
                outputs_val = model(input_ids=x_val_inputs,
                                    attention_mask=x_val_masks,
                                    lm_labels=lm_labels,
                                    decoder_attention_mask=y_val_masks)
                loss = outputs_val[0]
                running_loss += loss.item()

                # pred_ids = model.t5_model.generate(input_ids=x_val_inputs,
                #                                    attention_mask=x_val_masks)

                # b_y_pred = predict_t5(model, val_dataset)

                # print(tokenizer.decode(pred_ids[0]))
                # y_pred = list(itertools.chain(y_pred, b_y_pred))
                # y_true = list(itertools.chain(y_true, y_labels))

        # y_pred = predict_t5(model, val_dataset)
        # calculate validation loss
        val_loss = running_loss / len(val_loader)  # calculate validation loss

        # print status
        if show_progress:
            print('\n Epoch = %d, Train loss = %.5f, Val loss = %.5f' % (epoch_idx, train_loss, val_loss))

        # append training and validation loss
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model = model
        else:
            break

        pbar.reset()
    # save best model
    if save_path is not None:
        save_path_name = save_path + 'epoch_{}_{:.5f}.pt'.format(epoch_idx, val_loss)
        torch.save(best_model, save_path_name)

    return train_losses, val_losses


def train(model, train_dataset, val_dataset, epochs, lr, batch_size, show_progress=False, save_path=None):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Construct data loader from training and validation dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    val_losses = []
    train_losses = []

    num_train = len(train_dataset)

    # Training
    for epoch in range(epochs):
        # backprop
        running_loss = 0.0
        inner_iter = 0
        pbar = tqdm(train_loader, position=0, leave=True)
        epoch_idx = int(epoch + 1)
        model.train()
        for input_id, attention_mask, y in pbar:
            pbar.set_description("Processing Epoch %d" % epoch_idx)

            outputs = model(input_id, attention_mask)
            optimizer.zero_grad()
            # print(outputs.type(),y.type())
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # calculate training loss
        train_loss = running_loss / len(train_loader)  # calculate validation loss

        with torch.no_grad():
            running_loss = 0.0
            for input_id_val, attention_mask_val, y_val in val_loader:
                outputs_val = model(input_id_val, attention_mask_val)
                running_loss += criterion(outputs_val, y_val).item()

        # calculate validation loss
        val_loss = running_loss / len(val_loader)  # calculate validation loss

        # print status
        if show_progress:
            print('\n Epoch = %d, Train loss = %.5f, Val loss = %.5f' % (epoch_idx, train_loss, val_loss))

        # append training and validation loss
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        # save model at each epoch
        if save_path is not None:
            save_path_name = save_path + 'epoch_{}_{:.5f}.pt'.format(epoch_idx, val_loss)
            torch.save(model, save_path_name)
        pbar.reset()

    return train_losses, val_losses


def predict_t5(model, t5_dataset, batch_size=32):
    data_loader = DataLoader(t5_dataset, batch_size=batch_size)
    pred = []
    model.eval()
    for x_val_inputs, x_val_masks, _, _, _, _ in data_loader:
        with torch.no_grad():
            b_y_pred = model.t5_model.generate(input_ids=x_val_inputs, attention_mask=x_val_masks)
        pred += b_y_pred.tolist()
    output = []
    for p in pred:
        str_pred = config.TOKENIZER.decode(p)
        str_pred = str_pred.replace("<pad>", "").replace("</s>", "").strip()
        str_pred = re.split(', |: ', str_pred)
        if str_pred[0] == "goemo":
            try:
                output.append([str_to_num(config.GOEMO_MAPPING)[string] for string in str_pred[1:]])
            except:
                print("Printing unrecognized word in GoEmotions mapping: {}".format(str_pred))
                output.append([27])  # Append neutral if we cannot find the word
        elif str_pred[0] == "ekman":
            try:
                output.append([str_to_num(config.EKMAN_MAPPING)[string] for string in str_pred[1:]])
            except:
                print("Printing unrecognized word in Ekman mapping: {}".format(str_pred))
                output.append([6])  # Append neutral if we cannot find the word
        elif str_pred[0] == "senti":
            try:
                output.append([str_to_num(config.SENTI_MAPPING)[string] for string in str_pred[1:]])
            except:
                print("Printing unrecognized word in Sentiment mapping: {}".format(str_pred))
                output.append([3])  # Append neutral if we cannot find the word
        else:
            print("Using a different hyperparameter other than (goemo, ekman, senti)")
            print(str_pred)
    return label_multi_one_hot(output)


class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 7
        self.MODEL_PATH = 't5-base'

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LR = 5e-5
        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCELoss'
        self.EPOCHS = 1

        # data
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.BATCH_SIZE = 16
        self.TXT_MAX_LENGTH = 64
        self.TGT_MAX_LENGTH = 64
        self.EKMAN_JSON = 'data/ekman_mapping.json'
        self.SENTI_JSON = 'data/sentiment_mapping.json'
        self.EKMAN_MAPPING = json_mapping(self.EKMAN_JSON)
        self.EKMAN_MAPPING[6] = 'neutral'
        self.SENTI_MAPPING = json_mapping(self.SENTI_JSON)
        self.SENTI_MAPPING[3] = 'neutral'
        self.GOEMO_MAPPING = mapping()


config = Config()
