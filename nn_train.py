import torch
import torch.nn as nn
import numpy as np
import math



class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 1)


    def forward(self, x):
        y_predicted = torch.sigmoid(self.l1(x))
        return y_predicted




def learn_weights(inputs, outputs):
    X = torch.from_numpy(inputs.astype(np.float32))
    y = torch.from_numpy(outputs.astype(np.float32))
    y = y.view(len(outputs), 1)

    n_samples, n_features = X.shape
    input_size = n_features
    output_size = 1

    learning_rate = 0.01
    model = NeuralNet(input_size = input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    num_epochs = 500

    for epoch in range(num_epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            [w, b] = model.parameters()
            for i in range(len(w[0])):
                if w[0][i] < 0.0:
                    w[0][i] = 0.0

    with torch.no_grad():
        [w, b] = model.parameters()
        return w.detach().numpy()[0]
