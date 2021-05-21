import time
import utils
from models import NNModel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from gensim.models import KeyedVectors


# hyper-parameters
EPOCHS = 20
BATCH_SIZE = 32

LR = 0.001
KERNEL_CONV = 8  # 8
DROP_PROB = 0.2
OUTPUT_CONV = 256  # 512
LAMBDA = 20


# load description data
desc_data = pd.read_csv("data/desc_data.csv", header=None)
desc_data = desc_data[1].apply(lambda x: x.split(' '))
desc_data = desc_data.apply(lambda x: [int(xx) for xx in x]).values
desc_data = [torch.tensor(xx) for xx in desc_data]
max_desc_len = max([len(dd) for dd in desc_data])
desc_data = utils.pad_x(desc_data, max_len=max_desc_len)
desc_data = torch.tensor(desc_data)


# load dataset
train_loader = utils.load_data(BATCH_SIZE, label='train', desc_data=desc_data)
valid_loader = utils.load_data(BATCH_SIZE, label='valid', desc_data=desc_data)
test_loader = utils.load_data(BATCH_SIZE, label='test', desc_data=desc_data)


# word embedding matrix
we = KeyedVectors.load("model/word2vec50.wordvectors")
wordEmbeddingsMatrix = we.vectors
wordEmbeddingsMatrix = torch.FloatTensor(wordEmbeddingsMatrix)
wordEmbeddingsMatrix = torch.cat((torch.zeros((1, 100)), wordEmbeddingsMatrix))
del we


# initialize training model
model = NNModel(50,
                wordEmbeddingsMatrix,
                output_conv=OUTPUT_CONV,
                k_conv=KERNEL_CONV,
                dropout=DROP_PROB,
                lmbda=LAMBDA)
optimizer = optim.Adam(model.parameters(), lr=LR)


# training
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
best_val_acc = 0.0
best_model = None

for epoch in range(EPOCHS):
    train_accuracy, train_loss = utils.train(model, train_loader, optimizer=optimizer, epoch=epoch)
    valid_loss, valid_accuracy, _ = utils.evaluate(model, valid_loader)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    is_best = valid_accuracy > best_val_acc
    if is_best:
        best_val_acc = valid_accuracy
        best_model = model


# save best model to local
model_name = f'best_model_{int(time.time())}.pth'
model_path = f'model/{model_name}'
torch.save(best_model, model_path)


# load best model for testing
best_model = torch.load(model_path)
test_loss, test_accuracy, results = utils.evaluate(best_model, test_loader)


# print evaluation metrics
test_ytrue = np.array([x[0] for x in results])
test_yhat = np.array([x[1] for x in results])

test_auc, test_f1 , test_accuracy2 = utils.eval_score(test_ytrue, test_yhat)
print(f'''
        Test AUC: {test_auc},
        Test f1: {test_f1},
        Test accuracy: {test_accuracy2},
        Test accuracy-0: {test_accuracy}''')


# save model history
with open('model/tune_history.txt', 'a') as f:
    f.write(f'''
            ACC: {test_accuracy} | 
            AUC: {test_auc} | 
            F1: {test_f1} | 
            LR: {LR} | 
            LAMBDA: {LAMBDA} | 
            CONVOLUTION: kernel size {KERNEL_CONV}, output size {OUTPUT_CONV} | 
            DROPOUT: {DROP_PROB} | 
            EPOCHS: {EPOCHS} |
            BATCH: {BATCH_SIZE}
            ''')


# plot training curves
plt.plot(train_accuracies, label='accuracy')
plt.plot(train_losses, label='loss')
plt.title('Accuracy and loss during training')
plt.legend()
plt.savefig('train_acc_loss_fig.png')
plt.show()

plt.plot(valid_accuracies, label='accuracy')
plt.plot(valid_losses, label='loss')
plt.title('Accuracy and loss during validation')
plt.legend()
plt.savefig('valid_acc_loss_fig.png')
plt.show()
