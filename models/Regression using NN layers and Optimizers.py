import torch
import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([[4.7], [2.4], [7.5], [7.1], [4.3], [7.816], [8.9], [5.2], [8.59], [2.1], [8], [10],[4.5], [6], [4]], dtype = np.float32)

y_train = np.array([[2.6], [1.6], [3.09], [2.4], [2.4], [3.357], [2.6], [1.96], [3.53], [1.76], [3.2], [3.5], [1.6], [2.5], [2.2]], dtype = np.float32)

plt.figure(figsize = (12, 8))
plt.scatter(x_train, y_train, label = 'Original data', s = 250, c = 'g')
plt.legend()
plt.show()

x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)
x.size()

inp = 1
hidden = 5
out = 1

model = torch.nn.Sequential(torch.nn.Linear(inp, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, out))

loss_fn = torch.nn.MSELoss(reduction = 'sum')
learning_rate = 1e-4
for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            
predicted_in_tensor = model(x)
predicted = predicted_in_tensor.detach().numpy()

plt.figure(figsize = (12, 8))
plt.scatter(x_train, y_train, label = 'Original data', s = 250, c = 'g')
plt.plot(x_train, predicted, label = 'Fitted line')
plt.legend()
plt.show()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
predicted_in_tensor = model(x)
predicted = predicted_in_tensor.detach().numpy()

plt.figure(figsize = (12, 8))
plt.scatter(x_train, y_train, label = 'Original data', s = 250, c = 'g')
plt.plot(x_train, predicted, label = 'Fitted line')
plt.legend()
plt.show()
