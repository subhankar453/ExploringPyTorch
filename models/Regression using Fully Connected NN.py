import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

data = pd.read_csv('datasets/bike_sharing_day.csv', index_col = 0)
data.shape

plt.figure(figsize = (8, 6))
sns.barplot('yr', 'cnt', hue = 'season', data = data, ci = None)
plt.legend(loc = 'upper right', bbox_to_anchor = (1.2, 0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented')
plt.title('Number of bikes rented per season')

plt.figure(figsize = (8, 6))
sns.barplot(x = 'mnth', y = 'cnt', hue = 'workingday', data = data)
plt.title('Number of bikes rented per month')
plt.show()

plt.figure(figsize = (8, 6))
sns.scatterplot(x = 'temp', y = 'cnt', data = data)
plt.xlabel('Temperature')
plt.ylabel('Total number of bikes rented')

columns = ['season', 'registered', 'temp', 'atemp', 'holiday', 'weekday', 'weathersit', 'yr', 'mnth', 'workingday']
features = data[columns]
target = data[['cnt']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)
x_train_tensor = torch.tensor(x_train.values, dtype = torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype = torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float)

import torch.utils.data as data_utils
train_data = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = data_utils.DataLoader(train_data, batch_size = 100, shuffle = True)
len(train_loader)
features_batch, target_batch = iter(train_loader).next()
features_batch.size()

input_size = x_train_tensor.shape[1]
output_size = 1
hidden_size = 10
loss_fn = torch.nn.MSELoss()

model = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                            torch.nn.ReLU(),  #torch.nn.Sigmoid()
                            torch.nn.Dropout(p = 0.2),
                            torch.nn.Linear(hidden_size, output_size))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

total_step = len(train_loader)
num_epochs = 10000
for epoch in range(num_epochs + 1):
    for i, (features, target) in enumerate(train_loader):
        output = model(features)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
model.eval()
    
sample = x_test.iloc[45]
sample_tensor = torch.tensor(sample.values, dtype = torch.float)
with torch.no_grad():
    y_pred = model(sample_tensor)
print('Predicted count->', y_pred.item())
print('Actual count->', y_test.iloc[45])

with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()
compare_df = pd.DataFrame({'actual' : np.squeeze(y_test.values), 'predcted' : np.squeeze(y_pred)})
sklearn.metrics.r2_score(y_test, y_pred)

plt.figure(figsize = (8, 8))
plt.scatter(y_pred, y_test.values, s = 250)
plt.xlabel('Actual count')
plt.ylabel('predicted Count')
plt.show()