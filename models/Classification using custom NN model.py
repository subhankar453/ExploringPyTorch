import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('datasets/admission_predict.csv')
data.shape
data.describe()
data = data.rename(index=str, columns = {'Chance of Admit ' : 'Admit_Probability'})
#data = data[['GRE Score', 'TOEFL Score', 'University Rating','SOP', 'LOR ', 'CGPA', 'Research', 'Admit_Probability']]
data = data.drop('Serial No.', axis = 1)

plt.figure(figsize = (8, 8))
fig = sns.regplot(x = 'GRE Score', y = 'TOEFL Score', data = data)
plt.title('GRE Score bs TOEFL Score')
plt.show()

plt.figure(figsize = (8, 8))
fig = sns.regplot(x = 'GRE Score', y = 'CGPA', data = data)
plt.title('GRE Score bs CGPA')
plt.show()

plt.figure(figsize = (8, 8))
fig = sns.scatterplot(x = 'Admit_Probability', y = 'CGPA', data = data, hue = 'Research')
plt.title('CGPA vs Admit_Probability')
plt.xlabel('Admit_Probability')
plt.ylabel('CGPA')

plt.figure(figsize = (10, 10))
sns.heatmap(data.corr(), annot = True, linewidth = 0.05, fmt = '.2f')
plt.show()

from sklearn import preprocessing
data[['GRE Score', 'TOEFL Score', 'SOP', 'LOR ', 'CGPA']] = preprocessing.scale(data[['GRE Score',
                                                                                      'TOEFL Score',
                                                                                      'SOP',
                                                                                      'LOR ',
                                                                                      'CGPA']])




col = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR ', 'CGPA']
features = data[col]                                                                   
target = data[['Admit_Probability']]
y = target.copy()                                                                                  
y.replace(to_replace = target[target >= 0.8], value = int(2), inplace = True)
y.replace(to_replace = target[target >= 0.6], value = int(1), inplace = True)  
y.replace(to_replace = target[target < 0.6], value = int(0), inplace = True)  
target = y             

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)                                                                         
x_train.shape
y_train.shape
x_train_tensor = torch.from_numpy(x_train.values).float()
x_test_tensor = torch.from_numpy(x_test.values).float()
y_train_tensor = torch.from_numpy(y_train.values).view(1, -1)[0].long()
y_test_tensor = torch.from_numpy(y_test.values).view(1, -1)[0].long()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

input_size = x_train_tensor.shape[1]
output_size = len(target['Admit_Probability'].unique())

class Net(nn.Module):
    def __init__(self, hidden_size, activation_fn = 'relu', apply_dropout = False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.dropout = None
        if apply_dropout:
            self.dropout = nn.Dropout(0.2)
            
    def forward(self, x):
        activation_fn = None
        if self.activation_fn == 'sigmoid' :
            activation_fn = F.torch.sigmoid
        elif self.activation_fn == 'tanh' :
            activation_fn = F.torch.tanh
        elif self.activation_fn == 'relu' :
            activation_fn = F.relu
        x = activation_fn(self.fc1(x))
        x = activation_fn(self.fc2(x))
        if self.dropout != None:
            x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim = -1)

def train_and_evaluate_models(model, learn_rate = 0.001):
    epoch_data = []
    epochs = 1001
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)
    loss_fn = nn.NLLLoss()
    test_accuracy = 0.0
    for epoch in range(1, epochs):
        optimizer.zero_grad()
        model.train()
        y_pred = model(x_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        y_pred_test = model(x_test)
        loss_test = loss_fn(y_pred_test, y_test_tensor)
        _, pred = y_pred_test.data.max(1)
        test_accuracy = pred.eq(y_test.data).sum().item() / y_test.values.size
        epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), test_accuracy])
        if epoch % 100 == 0:
            print('epoch - %d (%d%%) train loss - %.2f test loss - %.2f Test accuracy - %.4f' % (epoch, epoch / 150 * 10,
                                                                                                 loss.data.item(),
                                                                                                 loss_test.data.item(),
                                                                                                 test_accuracy))
    return {'model->' : model,
            'epoch data->' : epoch_data,
            'num_epochs->' : epochs,
            'optimizer->' : optimizer,
            'loss_fn->' : loss_fn,
            'accuracy->' : test_accuracy,
            '_, pred' : y_pred_test.data.max(1),
            'actual_test_label' : y_test}

net = Net(hidden_size = 3, activation_fn = 'sigmoid', apply_dropout = False)
net
result_3_sigmoid = train_and_evaluate_models(net)
net = Net(hidden_size = 3, activation_fn = 'sigmoid', apply_dropout = True)
result_3_sigmoid = train_and_evaluate_models(net)
net = Net(hidden_size = 50, activation_fn = 'tanh')
result_50_tanh = train_and_evaluate_models(net)
net = Net(hidden_size = 1000, activation_fn = 'tanh')
result_1000_tanh = train_and_evaluate_models(net)
net = Net(hidden_size = 1000, activation_fn = 'tanh', apply_dropout = True)
result_1000_tanh = train_and_evaluate_models(net)
net = Net(hidden_size = 100, activation_fn = 'sigmoid', apply_dropout = False)
result_100_sigmoid = train_and_evaluate_models(net)
result_model = result_50_tanh
df_epochs_data = pd.DataFrame(result_model['epoch data'],
                              columns = ['epoch', 'train_loss', 'test_loss', 'accuracy'])

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8))
df_epochs_data[['train_loss', 'test_loss', ]].plot(ax = ax1)
df_epochs_data[['accuracy']].plot(ax = ax2)
plt.ylim(bottom = 0.5)
plt.show()


