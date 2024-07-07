from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


def resultAnalysis(result, test_y):
    misclassified_indices = [i for i in range(len(result)) if result[i] != test_y[i]]

    print("scikit-learn MLPClassifier accuracy: ", accuracy_score(result, test_y))
    print("misclassified: ", len(misclassified_indices), " out of: ", len(test_y))


def sklearnMLP():
    """ SIMPLER DATASET OF 8X8 HANDWRITTEN DIGITS BUILT IN SKLEANR
        data = load_digits(n_class=10, return_X_y=False, as_frame=False)
        X = list(zip(data['data'], data['images']))
        y = data.target

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=1)
        train_x_data = [item[0] for item in train_x]
        train_x_images = [item[1] for item in train_x]
        test_x_data = [item[0] for item in test_x]
        test_x_images = [item[1] for item in test_x]
        """

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor(), )
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor(), )
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=len(train_data))
    test_dataloader = DataLoader(train_data, batch_size=len(test_data))
    X, y = next(iter(train_dataloader))
    x_train_numpy = X.numpy()
    y_train_numpy = y.numpy()
    x_train_numpy = [item.flatten() for item in x_train_numpy]
    X_t, y_t = next(iter(test_dataloader))
    x_test_numpy = X_t.numpy()
    y_test_numpy = y_t.numpy()
    x_test_numpy = [item.flatten() for item in x_test_numpy]

    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(16, 16), batch_size=batch_size,
                        max_iter=5)

    clf.fit(x_train_numpy, y_train_numpy)
    result = clf.predict(x_test_numpy)

    resultAnalysis(result, y_test_numpy)


def pytorchMLP(training=True):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
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

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor(),)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor(),)
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(train_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to("cpu")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if training:
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("Done!")

        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")
    else:
        model = NeuralNetwork().to("cpu")
        model.load_state_dict(torch.load("model.pth"))

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to("cpu")
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
            img = x.squeeze()
            plt.imshow(img, cmap="gray")
            plt.show()
            test(test_dataloader, model, loss_fn)

if __name__ == '__main__':
    sklearnMLP()
    # pytorchMLP(training=False)
