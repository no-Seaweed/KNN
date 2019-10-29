import numpy as np


class KNeighborsClassifier:
    def __init__(self, k):
        self.k = k

    # def fit(self, X_train, y_train):
        # np.sqrt(np.sum(np.power(, 2)))
        # raise NotImplementedError()

    def predict(self, X_train, X_test, y_train):
        # raise NotImplementedError()
        train = np.concatenate((X_train, y_train.reshape(100, 1)), axis=1)
        distance_store = []
        neighbor = []
        for i in range(X_test.shape[0]):
            for j in range(X_train.shape[0]):
                euclidean_distance = np.sqrt(np.sum((X_test[i, :]-X_train[j, :])**2))
                distance_store.append(euclidean_distance)
            concat_temp = np.concatenate((np.array(distance_store).reshape(100, 1), y_train.reshape(100, 1)), axis=1)
            dist = concat_temp[concat_temp[:, 0].argsort()][:self.k, -1]
            distance_store = []
            neighbor.append(dist)
            dist = []
        # print(np.array(neighbor))
        neighbors = np.array(neighbor)
        # print(neighbors.shape)
        y_pred = []
        for neighbor in neighbors:
            (values, counts) = np.unique(neighbor, return_counts=True)
            ind = np.argmax(counts)
            y_pred.append(int(values[ind]))
        return np.array(y_pred)

    def score(self, y_pred_test, y_test):
        # raise NotImplementedError()
        return float(sum(y_pred_test == y_test)) / float(len(y_test))
