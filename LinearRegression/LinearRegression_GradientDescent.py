import numpy as np
import pandas as pd

class LinearRegression:
    """
    A class to use linear regression to fit a linear model to minimize the error.

    ...

    Attributes
    ----------
    learning_rate : float
        Determines the step size at each iteration while moving toward a minimum of a loss function.
    nb_iters : int
        The number of iterations to find the weights.
    weights : array of float
        Tells the software how important the feature should be in the model fit.
    bias : float
        The tendency of the regression result to land consistently offset from the origin.

    Methods
    -------
    fit():
        Train the linear model
    predict():
        Used for inference
    
    """
    def __init__(self, learning_rate = 0.0001, nb_iters = 1000):
        """
        Constructs all the necessary attributes for the LinearRegression object.

        Parameters
        ----------
            learning_rate : float
                Determines the step size at each iteration while moving toward a minimum of a loss function.
            nb_iters : int
                The number of iterations to find the weights.
        """

        self.learning_rate = learning_rate
        self.nb_iters = nb_iters
        # will initialize it in the fit function as we don't know the number of features
        self.weights = None
        self.bias = 0

    def gradient_descent(self, X, y, nb_samples):
        """
        Finds the best-fit line for a given training dataset

        Parameters
        ----------
            X : array of float
                The Dataset that we want to train with.
            y : array of float
                The true values of each sample.
            nb_samples : 
                The total number of samples thaat are being used for trainning.
        """

        y_pred = np.dot(self.weights, np.transpose(X)) + self.bias
        
        # Gradients
        dw = (1/nb_samples) * np.dot(np.transpose(X), (y_pred - y))
        db = (1/nb_samples) * np.sum((y_pred - y))

        # Update the Weights and bias
        self.weights = self.weights - (self.learning_rate * dw)
        self.bias = self.bias - (self.learning_rate * db)

    def fit(self, X, y):
        """
        Train the linear model using Gradient Descent.

        Parameters
        ----------
            X : array of float
                The Dataset that we want to train with.
            y : array of float
                The true values of each sample.
        """

        X = pd.DataFrame(X)
        nb_samples, nb_features = X.shape

        self.weights = np.zeros(nb_features)
        self.bias = 0

        ''' use Gradient Descent '''

        for i in range(self.nb_iters):
            self.gradient_descent(X, y, nb_samples)

    def predict(self, X):
        """
        Predict the values of the target for given data.

        Parameters
        ----------
            X : array of float
                The Dataset that we want to predict the value of the target for.
        """

        X = pd.DataFrame(X)
        y_pred = np.dot(self.weights, np.transpose(X)) + self.bias

        return y_pred