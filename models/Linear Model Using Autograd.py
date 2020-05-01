import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = np.array([[4.7], [2.4], [7.5], [7.1], [4.3], [7.816], [8.9], [5.2], [8.59], [2.1], [8], [10],[4.5], [6], [4]], dtype = np.float32)

y_train = np.array([[2.6], [1.6], [3.09], [2.4], [2.4], [3.357], [2.6], [1.96], [3.53], [1.76], [3.2], [3.5], [1.6], [2.5], [2.2]], dtype = np.float32)

plt.figure(figsize = (12, 8))
plt.scatter(x_train, y_train, label = 'Original data', s = 250, c = 'g')
plt.legend()
plt.show()

import torch
X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)
input_size = 1
hidden_size = 1
output_size = 1

w1 = torch.rand(input_size, hidden_size, requires_grad = True)
w1.shape
w2 = torch.rand(hidden_size, output_size, requires_grad = True)
learning_rate = 1e-6
for iter in range(1, 10000):
    y_pred = X_train.mm(w1).mm(w2)
    loss = (y_pred - Y_train).pow(2).sum()
    if iter % 50 == 0:
        print(iter, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

print('w1->', w1)
print('w2->', w2)

x_train_tensor = torch.from_numpy(x_train)
predicted_in_tensor = x_train_tensor.mm(w1).mm(w2)
predicted = predicted_in_tensor.detach().numpy()
predicted

plt.figure(figsize = (12, 8))
plt.scatter(x_train, y_train, label = 'Original data', s = 250, c = 'g')
plt.plot(x_train, predicted, label = 'Fitted line')
plt.legend()
plt.show()