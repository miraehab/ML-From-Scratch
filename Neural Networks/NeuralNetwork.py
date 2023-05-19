import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def feed_forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backpropagation(self, output_gradient, learning_rate):
        # output_gradient = dE/dY 
        # dE/dW = (dE/dY)*X.T
        dw = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate*dw
        # dE/dB = dE/dY(output_gradient)
        self.bias -= learning_rate*output_gradient
        # dE/dX = W.T*(dE/dY)(output_gradient)
        return np.dot(self.weights.T, output_gradient)

def sigmoid(linear_pred):
    return (1 / (1 + np.exp(-linear_pred)))

def mse(y_true, y_pred):
    squared_error = np.square(np.subtract(y_true, y_pred))

    return squared_error.mean()

def mse_prime(y_true, y_pred):
    return(2*(y_pred-y_true)/y_true.size);

class NN:
    def __init__(self, y, num_of_layers, size_of_layers, epochs = 10000, learning_rate = 0.1):
        self.num_of_layers = num_of_layers
        self.size_of_layers = size_of_layers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self, X, y):
        network = []
        last_output = X.shape[0]
        for i in range(self.num_of_layers):
            # We initialize every layer with the input size which is equal to the output of the previous layer
            network.append(Layer(last_output, self.num_of_layers[i]))

        for i in range(self.epochs):
            error = 0
            for x, y in zip(X, y):
                ### Feed Forward ###
                output = x
                for layer in network:
                    output = sigmoid(layer.feed_forward(output))

                # Calculate the error for each sample
                error += mse(y, output)

                ### Backpropagation ###
                output_gradient = mse_prime(y, output)
                for layer in reversed(network):
                    output_gradient = sigmoid(layer.backpropagation(output_gradient, self.learning_rate))

            error /= X.shape[0]

            print("{}/{} loss: {} ".format(i, self.epochs, error))

