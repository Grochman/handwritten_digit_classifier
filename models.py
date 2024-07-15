import numpy as np
import math
import pickle

import torch
from torch import nn


def relu(x):
    return np.maximum(x, 0)


def relu_dir(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def one_hot(x):
    encoding = np.zeros((len(x), 10))
    for i in range(len(x)):
        encoding[i, x[i]] = 1
    return encoding.T


def decode(x):
    return np.argmax(x, 0)


def crossEntrpyLoss(x, y):
    return -np.sum(y * np.log(x))


class FromScratchModel:
    def __init__(self, lr=0.1):
        self.layers = (28 * 28, 256, 64, 10)
        self.w = [np.random.rand(self.layers[i + 1], self.layers[i]) - 0.5 for i in range(len(self.layers) - 1)]
        self.b = [np.random.rand(self.layers[i + 1], 1) - 0.5 for i in range(len(self.layers) - 1)]
        self.a = [np.zeros((i, 1)) for i in self.layers]
        self.z = [np.zeros((i, 1)) for i in self.layers]
        self.lr = lr

    def forward(self, x):
        self.a[0] = np.array(x)
        for i in range(len(self.w) - 1):
            self.z[i + 1] = self.w[i].dot(self.a[i]) + self.b[i]
            self.a[i + 1] = relu(self.z[i + 1])
        self.z[-1] = self.w[-1].dot(self.a[-2]) + self.b[-1]
        self.a[-1] = softmax(self.z[-1])
        return self.a[-1]

    def backprop(self, pred, y):
        m = len(y)
        y = one_hot(y)
        dw = [0] * len(self.b)
        db = [0] * len(self.b)
        delta = [0] * (len(self.b) + 1)
        delta[-1] = pred - y
        for i in range(len(self.w) - 1, -1, -1):
            dw[i] = 1 / m * delta[i + 1].dot(self.a[i].T)
            db[i] = 1 / m * np.sum(delta[i])
            delta[i] = self.w[i].T.dot(delta[i + 1]) * relu_dir(self.z[i])

        for i in range(len(self.w)):
            self.w[i] -= self.lr * dw[i]
            self.b[i] -= self.lr * db[i]

    def train(self, x, y, batch_size, iterations=5):
        for i in range(iterations):
            for j in range(math.ceil(x.shape[0] / batch_size)):
                start = batch_size * j
                end = min(batch_size * (j + 1), x.shape[0])
                self.backprop(self.forward(x[:, start:end]), y[start:end])
            print("iteration: ", i + 1)

    def load(self):
        with open('from_scratch_model_w.pkl', 'rb') as file:
            self.w = pickle.load(file)
        with open('from_scratch_model_b.pkl', 'rb') as file:
            self.b = pickle.load(file)

    def load_torch(self):
        model_wb = torch.load("pytorch_model.pth")
        numpy_arrays = [tensor.numpy() for tensor in model_wb.values()]
        self.w[0] = numpy_arrays[0]
        self.w[1] = numpy_arrays[2]
        self.w[2] = numpy_arrays[4]
        self.b[0] = numpy_arrays[1].reshape(-1, 1)
        self.b[1] = numpy_arrays[3].reshape(-1, 1)
        self.b[2] = numpy_arrays[5].reshape(-1, 1)


class PytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, x, device):
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            pred = self(x)
            predicted = classes[pred[0].argmax(0)]
        return predicted


