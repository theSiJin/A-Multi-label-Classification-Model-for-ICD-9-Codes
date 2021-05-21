import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import roc_auc_score


"""
Data Preprocessing
"""


# transform label series to y-like array
def label_to_y(df, dict_labels):
    m, n = df.shape
    y = np.zeros((m, len(dict_labels)))
    for i in range(m):
        labels = df.iloc[i, 5]
        for l in labels:
            idx = dict_labels[l]
            y[i, idx] = 1
    return y


# add padding 0
def pad_x(x, max_len=2500):
    result = np.zeros((len(x), max_len), dtype=int)
    i = 0
    for ids in x:
        result[i, :len(ids)] = ids
        i += 1
    return result


"""
ICD-9 Code Descriptions
"""


# create dict of code descriptions
def code_map_desc(codes):
    desc_dict = {}
    with open('data/ICD_descriptions', 'r') as f:
        for i, row in enumerate(f):
            row = row.strip().split()
            cd = row[0]
            if cd in codes:
                desc_dict[cd] = ' '.join(row[1:])

    return desc_dict


"""
Word Embeddings
"""


# create a pre-trained embedding matrix
def word_embeddings(col_tokens, de=100, file_name='word2vec50.wordvectors'):
    model = Word2Vec(min_count=0, size=de)
    model.build_vocab(col_tokens)
    model.train(col_tokens, total_examples=model.corpus_count, epochs=model.epochs)

    model.wv.save(file_name)
    we = KeyedVectors.load(file_name)

    return torch.FloatTensor(we.vectors)


"""
Load Tensor Data
"""


# load data from local directory
def load_data(batch_size, label='train', desc_data=None):
    X = torch.load('data/for_training/X_' + label + '50.pt')
    y = torch.load('data/for_training/y_' + label + '50.pt')
    if desc_data is not None:
        d = torch.cat(X.shape[0] * [desc_data]).view(X.shape[0], len(desc_data),-1)
        ds = torch.utils.data.TensorDataset(X, y, d)
    else:
        ds = torch.utils.data.TensorDataset(X, y)

    if label == 'train':
        dloader = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
    else:
        dloader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)
    
    return dloader


"""
Model Training and Evaluation 
"""


# train the model and print metrics
def train(model, data_loader, optimizer, epoch, print_freq=50):

    accuracy_list, auc_list, f1_list = [], [] ,[]
    loss_list = []

    model.train()
    sigmoid = nn.Sigmoid()
    for i, (input, target, desc_data) in enumerate(data_loader):

        optimizer.zero_grad()
        output, loss = model(input, target, desc_data=desc_data)
        loss.backward()
        optimizer.step()
        
        output = sigmoid(output)
        yhat = (output > 0.5).int()
        yhat = yhat.detach().numpy()
        y = target.detach().numpy()

        auc, f1, acc = eval_score(yhat, y)

        loss_list.append(loss.item())
        accuracy_list.append(auc)
        auc_list.append(auc)
        f1_list.append(f1)

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}]\t',
                  f'Loss: {loss.item():.4f}\t',
                  f'AUC: {auc:.4f}\t',
                  f'Micro F1: {f1:.4f}\t',
                  f'Micro Accuracy: {acc:.4f}')

    return np.mean(accuracy_list), np.mean(loss_list)


# evaluate the model and print metrics
def evaluate(model, data_loader, print_freq=50):

    accuracy_list, auc_list, f1_list = [], [], []
    loss_list = []
    results = []
    
    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for i, (input, target, desc_data) in enumerate(data_loader):

            output, loss = model(input, target, desc_data=desc_data)
            
            output = sigmoid(output)
            yhat = (output > 0.5).int()
            yhat = yhat.detach().numpy()
            y = target.detach().numpy()
            
            auc, f1, acc = eval_score(yhat, y)

            loss_list.append(loss.item())
            accuracy_list.append(auc)
            auc_list.append(auc)
            f1_list.append(f1)
            
            results.extend(list(zip(y, yhat)))

            if i % print_freq == 0:
                print(f'Validation: [{i}/{len(data_loader)}]\t',
                      f'Loss: {loss.item():.4f}\t',
                      f'AUC: {auc:.4f}\t',
                      f'Micro F1: {f1:.4f}\t',
                      f'Micro Accuracy: {acc:.4f}')

    return np.mean(accuracy_list), np.mean(loss_list), results


"""
Evaluation Metrics
"""


# print all evaluation metrics
def eval_score(yhat, y, type='micro'):
    auc = auc_score(yhat, y, type=type)
    if type == 'micro':
        f1 = micro_f1(yhat.ravel(), y.ravel())
        acc = micro_accuracy(yhat.ravel(), y.ravel())
    else:
        f1, acc = 0, 0

    return auc, f1, acc.mean()


# auc score from scikit-learn
def auc_score(yhat, y, type='micro'):
    return roc_auc_score(y, yhat, multi_class='ovo', average=type)


# micro-averaged accuracy score
def micro_accuracy(yhat, y):
    if union_size(yhat, y, 0) == 0:
        return 0
    else:
        return intersect_size(yhat, y, 0) / union_size(yhat, y, 0)



def micro_precision(yhat, y):
    if yhat.sum(axis=0) == 0:
        return 0
    else:
        return intersect_size(yhat, y, 0) / yhat.sum(axis=0)


def micro_recall(yhat, y):
    if y.sum(axis=0) == 0:
        return 0
    else:
        return intersect_size(yhat, y, 0) / y.sum(axis=0)


# micro-averaged f1 score
def micro_f1(yhat, y):
    prec = micro_precision(yhat, y)
    rec = micro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec)/(prec + rec)
    return f1


def union_size(yhat, y, axis):
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)