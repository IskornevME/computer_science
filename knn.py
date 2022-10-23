class KNearestNeighbor:
    """
    a kNN classifier with L2 distance
    """
    
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(sum((X[i] - self.X_train[j]) ** 2))
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt((np.sum((X[i] - self.X_train)**2, axis = 1)))
	return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(np.sum(X ** 2, axis = 1)[:, np.newaxis] -\
        2*X.dot(self.X_train.T) + np.sum(self.X_train ** 2, axis = 1))
        return dists
