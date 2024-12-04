import numpy as np

class LogisticRegression:
    """
    Logistic Regression with Regularization and Improved Training Process.
    """
    def __init__(self, learning_rate, num_iter, add_bias, verbose, reg_strength, class_weight):
        """
        Parameters:
        - learning_rate: Learning rate for gradient descent.
        - num_iter: Number of iterations for training.
        - add_bias: Whether to add a bias term to the features.
        - penalty: Regularization type ('l1' or 'l2').
        - C: Inverse of regularization strength.
        - verbose: Whether to print loss during training.
        - reg_strength: Regularization strength.
        - class_weight: Dictionary with weights for each class (e.g., {0: 1, 1: 2}).
        """
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.add_bias = add_bias
        self.verbose = verbose
        self.reg_strength = reg_strength  # Regularization strength
        self.class_weight = class_weight

    def __add_bias(self, X):
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        bce_loss = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        reg_term = (self.reg_strength / (2 * y.size)) * np.sum(np.square(self.theta[1:]))
        return  bce_loss + reg_term

    def train(self, X, y):
        if self.add_bias:
            X = self.__add_bias(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)

            # Adjust gradients using class weights
            if self.class_weight:
                weights = np.where(y == 1, self.class_weight[1], self.class_weight[0])
                gradient = np.dot(X.T, weights * (y - h)) / y.size
            else:
                gradient = np.dot(X.T, (y - h)) / y.size

            # Apply regularization
            gradient[1:] += (self.reg_strength / y.size) * self.theta[1:]
            
            # Update weights
            self.theta += self.learning_rate * gradient

            if self.verbose and i % (self.num_iter // 10) == 0:
                loss = self.__loss(h, y)
                print(f"Iteration {i}: loss = {loss:.4f}")

    def predict_prob(self, X):
        if self.add_bias:
            X = self.__add_bias(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.35):
        prob = self.predict_prob(X)
        return (prob >= threshold).astype(np.int8)

    def evaluate(self, y_true: list, y_pred: list):
        """
        This function evaluates the performance of a binary classification model.

        Parameters:
        - y_true (list): A list of actual values.
        - y_pred (list): A list of predicted values by the model.

        Returns:
        - precision: the precision of the model
        - recall: the recall of the model
        - accuracy: the accuracy of the model
        """
        # Initialize variables
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        # Calculate true positives, false positives, false negatives, and true negatives
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                TN += 1

        # Calculate precision, recall and accuracy
        precision = 0.0 if (TP + FP) == 0 else (TP / (TP + FP))
        recall = 0.0 if (TP + FN) == 0 else (TP / (TP + FN))
        accuracy = (TP + TN) / len(y_true)
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return precision, recall, accuracy, f1_score
