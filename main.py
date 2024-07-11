from sklearn.neural_network import MLPClassifier
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np


def sklearnMLP(x_train, y_train, x_test, y_test, training=True):
    # reset default parameters for better comparison with other implementations
    clf = MLPClassifier(solver='sgd', alpha=0, hidden_layer_sizes=(16, 16), batch_size=64,
                        learning_rate_init=0.01, max_iter=5, shuffle=False, verbose=True, momentum=0)

    if training:
        clf.fit(x_train, y_train)
        joblib.dump(clf, "sklearn_model.pkl")
    else:
        clf = joblib.load("sklearn_model.pkl")
    print(f'sklearnMLP accuracy: {clf.score(x_test, y_test)}')


def pytorchMLP(train_dataloader, test_dataloader, training=True):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(device)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if training:
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            # test(test_dataloader, model, loss_fn)
        print("Done!")
        test(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), "pytorch_model.pth")
        print("Saved PyTorch Model State to pytorch_model.pth")
    else:
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load("pytorch_model.pth"))

        test(test_dataloader, model, loss_fn)

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'First predicted element: "{predicted}", Actual: "{actual}"')
            img = x.squeeze()
            plt.imshow(img, cmap="gray")
            plt.show()


def fromScratchMLP(x_train, y_train, x_test, y_test, training=True):
    def relu(x):
        return [max(0, x_i) for x_i in x]

    def relu_dir(x):
        return np.where(x > 0, 1, 0)

    def mse(x, y):
        return 0.5*np.sum((x-y)**2)

    def crossEntrpyLoss(x, y):
        return -np.sum(y * np.log(x + 1e-8))

    class Model:
        def __init__(self):
            self.layers = (28*28, 16, 16, 10)
            self.weights = [np.random.rand(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
            self.bias = [np.zeros(self.layers[i + 1]) for i in range(len(self.layers) - 1)]
            self.neuron_values = [np.zeros(i) for i in self.layers]
            self.pre_activation_values = [np.zeros(i) for i in self.layers]
            self.lr = 0.01

        def backprop(self, pred, y):
            dcdw = [0] * len(self.weights)
            delta = [0] * (len(self.bias)+1)
            delta[-1] = (pred - y)*relu_dir(self.pre_activation_values[-1])

            for i in range(len(self.weights) - 1, -1, -1):
                dcdw[i] = np.outer(self.pre_activation_values[i], delta[i+1])
                delta[i] = (delta[i+1] @ np.array(self.weights[i]).T) * relu_dir(self.pre_activation_values[i])

            for i in range(len(self.weights)):
                self.weights[i] -= dcdw[i]*self.lr
                self.bias[i] -= delta[i+1]*self.lr

        def forward(self, x):
            self.neuron_values[0] = x
            for i in range(len(self.weights)):
                self.pre_activation_values[i+1] = self.neuron_values[i] @ self.weights[i] + self.bias[i]
                self.neuron_values[i+1] = relu(self.pre_activation_values[i+1])
            return self.neuron_values[-1]

        def train(self, x, y):
            for i in range(len(x)):
                self.backprop(self.forward(x[i]), y[i])

    model = Model()

    model_wb = torch.load("pytorch_model.pth")
    numpy_arrays = [tensor.numpy() for tensor in model_wb.values()]
    model.weights[0] = numpy_arrays[0].T
    model.weights[1] = numpy_arrays[2].T
    model.weights[2] = numpy_arrays[4].T
    model.bias[0] = numpy_arrays[1]
    model.bias[1] = numpy_arrays[3]
    model.bias[2] = numpy_arrays[5]

    for i in range(len(x_test)):
        print(model.forward(x_test[i]), " ", y_test[i])


if __name__ == '__main__':

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor(), )
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor(), )
    train_dataloader = DataLoader(train_data, batch_size=len(train_data))
    test_dataloader = DataLoader(train_data, batch_size=len(test_data))
    X, y = next(iter(train_dataloader))
    x_train_np = X.numpy()
    y_train_np = y.numpy()
    x_train_np = [item.flatten() for item in x_train_np]
    X_t, y_t = next(iter(test_dataloader))
    x_test_np = X_t.numpy()
    y_test_np = y_t.numpy()
    x_test_np = [item.flatten() for item in x_test_np]

    batch_size = 64
    train_dataloader_batched = DataLoader(train_data, batch_size=batch_size)
    test_dataloader_batched = DataLoader(train_data, batch_size=batch_size)

    # sklearnMLP(x_train_np, y_train_np, x_test_np, y_test_np, training=False)
    # pytorchMLP(train_dataloader_batched, test_dataloader_batched, training=True)
    fromScratchMLP(x_train_np, y_train_np, x_test_np, y_test_np, training=True)
