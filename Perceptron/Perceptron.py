import numpy as np
import pandas as pd

def unit_step_function(x):
    return 1 if x > 0 else 0

class Perceptron:
    """
    A class to use perceptron algorithm for binary calssification problems.
    ...
    Attributes
    ----------
    learning_rate : float
        Determines the step size at each iteration while moving toward a minimum of a loss function.
    nb_iters : int
        The number of iterations to find the weights.
    activation_fun: function
        The function that maps the output to its class.
    weights : array of float
        Tells the software how important the feature should be in the model fit.
    bias : float
        The tendency of result to land consistently offset from the origin.
    Methods
    -------
    fit():
        Train the perceptron model
    predict():
        Used for inference
    
    """

    def __init__(self, learning_rate=0.01, nb_iters = 1000):
        """
        Constructs all the necessary attributes for the Perceptron object.
        Parameters
        ----------
            learning_rate : float
                Determines the step size at each iteration while moving toward a minimum of a loss function.
            nb_iters : int
                The number of iterations to find the weights.
        """

        self.learning_rate = learning_rate
        self.nb_iters = nb_iters
        self.activation_func = unit_step_function
        # we will initialize the weights in the fit function as we don't know the number of features
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        """
        Train the perceptron model.
        Parameters
        ----------
            X : array of float
                The Dataset that we want to train with.
            y : array of float
                The true values of each sample.
        """
         
        #X = pd.DataFrame(X)
        nb_samples, nb_features = X.shape

        # init weights
        # the number of features is the size of the input layer
        # we initialize the weights randomly because this helps in breaking symmetry and every neuron is no longer performing the same computation
        self.weights = np.random.rand(nb_features-1, nb_features)*0.01
        self.bias = 0

        for j in range(self.nb_iters):
            for i in range(nb_samples):
                linear_output = np.dot(X[i], np.transpose(self.weights)) + self.bias
                y_pred = self.activation_func(linear_output)

                # Update Weights and Bias
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)




    def predict(self, X):
        """
        Predict the values of the target for given instances.
        Parameters
        ----------
            X : array of float
                The Dataset that we want to predict the value of the target for.
        """
        linear_output = np.dot(X, np.transpose(self.weights)) + self.bias
        y_pred = []
        for i in linear_output:
            y_pred.append(self.activation_func(i))
        return np.array(y_pred)
