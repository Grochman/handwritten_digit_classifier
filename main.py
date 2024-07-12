from sklearn.neural_network import MLPClassifier

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np
import math
import pickle


def sklearnMLP(x_train, y_train, x_test, y_test, training=True):
    clf = MLPClassifier(solver='sgd', alpha=0, hidden_layer_sizes=(16, 16), batch_size=64,
                        learning_rate_init=0.01, max_iter=5, shuffle=False, verbose=True, momentum=0)

    if training:
        clf.fit(x_train, y_train)
        with open("sklearn_model.pkl", 'wb') as file:
            pickle.dump(clf, file, protocol=5)
    else:
        with open("sklearn_model.pkl", "rb") as file:
            clf = pickle.load(file)
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
    def load_pytorch_weights(model):
        model_wb = torch.load("pytorch_model.pth")
        numpy_arrays = [tensor.numpy() for tensor in model_wb.values()]
        model.w[0] = numpy_arrays[0]
        model.w[1] = numpy_arrays[2]
        model.w[2] = numpy_arrays[4]
        model.b[0] = numpy_arrays[1].reshape(-1, 1)
        model.b[1] = numpy_arrays[3].reshape(-1, 1)
        model.b[2] = numpy_arrays[5].reshape(-1, 1)

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

    class Model:
        def __init__(self, lr=0.1):
            self.layers = (28*28, 16, 16, 10)
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
            delta = [0] * (len(self.b)+1)
            delta[-1] = pred - y
            for i in range(len(self.w) - 1, -1, -1):
                dw[i] = 1/m * delta[i+1].dot(self.a[i].T)
                db[i] = 1/m * np.sum(delta[i])
                delta[i] = self.w[i].T.dot(delta[i + 1]) * relu_dir(self.z[i])

            for i in range(len(self.w)):
                self.w[i] -= self.lr * dw[i]
                self.b[i] -= self.lr * db[i]

        def train(self, x, y, batch_size, iterations=5):
            for i in range(iterations):
                for j in range(math.ceil(x.shape[0]/batch_size)):
                    start = batch_size*j
                    end = min(batch_size*(j+1), x.shape[0])
                    self.backprop(self.forward(x[:, start:end]), y[start:end])
                print("iteration: ", i+1)

    x_train = x_train.T
    x_test = x_test.T

    model = Model(lr=0.01)

    if training:
        model.train(x_train, y_train, 64, iterations=500)
        with open('from_scratch_model_w.pkl', 'wb') as file:
            pickle.dump(model.w, file)
        with open('from_scratch_model_b.pkl', 'wb') as file:
            pickle.dump(model.b, file)
    else:
        with open('from_scratch_model_w.pkl', 'rb') as file:
            model.w = pickle.load(file)
        with open('from_scratch_model_b.pkl', 'rb') as file:
            model.b = pickle.load(file)

    test_loss, correct = 0, 0
    size = len(y_test)
    for i in range(size):
        item = model.forward(x_test[:, i].reshape(-1, 1))
        if decode(item)[0] == y_test[i]:
            correct += 1
        test_loss += crossEntrpyLoss(item, one_hot([y_test[i]]))
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor(), )
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor(), )
    train_dataloader = DataLoader(train_data, batch_size=len(train_data))
    test_dataloader = DataLoader(train_data, batch_size=len(test_data))
    X, y = next(iter(train_dataloader))
    x_train_np = X.numpy()
    y_train_np = y.numpy()
    x_train_np = [item.flatten() for item in x_train_np]
    x_train_np = np.array(x_train_np)
    X_t, y_t = next(iter(test_dataloader))
    x_test_np = X_t.numpy()
    y_test_np = y_t.numpy()
    x_test_np = [item.flatten() for item in x_test_np]
    x_test_np = np.array(x_test_np)

    batch_size = 64
    train_dataloader_batched = DataLoader(train_data, batch_size=batch_size)
    test_dataloader_batched = DataLoader(train_data, batch_size=batch_size)

    # sklearnMLP(x_train_np, y_train_np, x_test_np, y_test_np, training=False)
    # pytorchMLP(train_dataloader_batched, test_dataloader_batched, training=True)
    fromScratchMLP(x_train_np, y_train_np, x_test_np, y_test_np, training=True)
