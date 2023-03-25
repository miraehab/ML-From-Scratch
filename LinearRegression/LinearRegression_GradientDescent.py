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
    weights : array of flloat
        Tells the software how important the feature should be in the model fit.
    bias : float
        The tendency of the regression result to land consistently offset from the origin.

    Methods
    -------
    
    """
    def __init__(self, learning_rate = 0.001, nb_iters = 1000):
        self.learning_rate = learning_rate
        self.nb_iters = nb_iters
        self.weights = None
        self.bias = None