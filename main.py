from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np


def sklearnMLP(x_train, y_train, x_test, y_test, training=True):
    """         SIMPLER DATASET OF 8X8 HANDWRITTEN DIGITS BUILT IN SKLEANR
    data = load_digits(n_class=10, return_X_y=False, as_frame=False)
    X = list(zip(data['data'], data['images']))
    y = data.target

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=1)
    train_x_data = [item[0] for item in train_x]
    train_x_images = [item[1] for item in train_x]
    test_x_data = [item[0] for item in test_x]
    test_x_images = [item[1] for item in test_x]
    """

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
            X, y = X.to("cpu"), y.to("cpu")

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
                X, y = X.to("cpu"), y.to("cpu")
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    model = NeuralNetwork().to("cpu")
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
        model = NeuralNetwork().to("cpu")
        model.load_state_dict(torch.load("pytorch_model.pth"))

        test(test_dataloader, model, loss_fn)

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to("cpu")
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
        return 0.5*sum([item**2 for item in np.subtract(x, y)])

    def crossEntrpyLoss(x, y):
        return -sum([y_i*np.log(x_i) for x_i, y_i in (x, y)])


    class Model:
        def __init__(self):
            self.layers = (5, 16, 16, 10)
            self.weights = [np.random.rand(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
            self.bias = [np.random.rand(self.layers[i + 1]) for i in range(len(self.layers) - 1)]
            self.neuron_values = [np.zeros(i) for i in self.layers]
            self.pre_activation_values = [np.zeros(i) for i in self.layers]
            self.lr = 0.01

        def backprop(self, pred, y):
            delta3 = (pred - y)*relu_dir(self.pre_activation_values[3]) #pred to to samo co ostatnie wyjscie
            dcdw3 = delta3@self.neuron_values[2].T

            delta2 = delta3 @ self.weights[2].T * relu_dir(self.pre_activation_values[2])
            dcdw2 = delta3@self.neuron_values[1].T

            #update weights
            self.weights[3] -= dcdw3*self.lr
            self.weights[2] -= dcdw2 * self.lr

        def forward(self, x):
            self.neuron_values[0] = x
            for i in range(len(self.weights)):
                self.pre_activation_values[i+1] = self.neuron_values[i] @ self.weights[i] + self.bias[i]
                self.neuron_values[i+1] = relu(self.pre_activation_values)
            return self.neuron_values[-1]

    model = Model()
    data = np.random.rand(5)
    result = model.forward(data)
    print(result)
    print("end")


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
