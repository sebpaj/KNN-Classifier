import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from visualize import Visualizer


def main():

    # create random training samples for 3 categories in range 0-3, 2-5, 4-7
    np.random.seed(5)
    X1 = [3 * np.random.random_sample(75) + 0, 3 * np.random.random_sample(75) + 0]
    X2 = [3 * np.random.random_sample(75) + 2, 3 * np.random.random_sample(75) + 2]
    X3 = [3 * np.random.random_sample(75) + 4, 3 * np.random.random_sample(75) + 4]
    X_train = np.hstack((X1, X2, X3)).T

    # create testing random samples in range 0-7
    X_test = np.array([7 * np.random.random_sample(30), 7 * np.random.random_sample(30)])

    # create labels for training data
    y1 = [1 for _ in range(75)]
    y2 = [2 for _ in range(75)]
    y3 = [3 for _ in range(75)]
    y_train = np.hstack((y1, y2, y3))

    # plot data in different colors
    plt.scatter(X1[0], X1[1], c='r', marker='s', label='X1')
    plt.scatter(X2[0], X2[1], c='b', marker='x', label='X2')
    plt.scatter(X3[0], X3[1], c='lightgreen', marker='o', label='X3')
    plt.scatter(X_test[0], X_test[1], c='black', marker='^', label='test set')
    plt.legend(loc='upper left')
    plt.show()

    # create classifier
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train, y_train)

    # prepare test set and predict labels
    X_test = X_test.T
    y_test = knn.predict(X_test)

    # combine train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    Visualizer.plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(225, 255))
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()