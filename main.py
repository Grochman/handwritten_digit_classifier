from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def resultAnalysis(result, test_y, test_x_images):
    misclassified_indices = [i for i in range(len(result)) if result[i] != test_y[i]]

    print("scikit-learn MLPClassifier accuracy: ", accuracy_score(result, test_y))
    print("missclassified: ", len(misclassified_indices), " out of: ", len(test_y))

    n_cols = 5
    n_rows = (len(misclassified_indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()
    for i, ax in zip(misclassified_indices, axes):
        ax.matshow(test_x_images[i], cmap='gray')
        ax.set_title(f"Pred: {result[i]}, Target: {test_y[i]}")
        ax.axis('off')

    for ax in axes[len(misclassified_indices):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def sklearnMLP():
    data = load_digits(n_class=10, return_X_y=False, as_frame=False)

    X = list(zip(data['data'], data['images']))
    y = data.target

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=1)
    train_x_data = [item[0] for item in train_x]
    train_x_images = [item[1] for item in train_x]
    test_x_data = [item[0] for item in test_x]
    test_x_images = [item[1] for item in test_x]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(16, 16),
                        max_iter=1_000_000)

    clf.fit(train_x_data, train_y)
    result = clf.predict(test_x_data)

    resultAnalysis(result,test_y,test_x_images)


if __name__ == '__main__':
    sklearnMLP()
