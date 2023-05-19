import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def feed_forward(self, input):
        self.input = input
          
        return np.dot(self.input, self.weights) + self.bias
    
    def backpropagation(self, output_gradient, learning_rate):
        # output_gradient = dE/dY 
        # dE/dW = (dE/dY)*X.T
        dw = np.dot(self.input.T, output_gradient)
        self.weights -= learning_rate*dw
        # dE/dB = dE/dY(output_gradient)
        self.bias -= learning_rate*output_gradient
        # dE/dX = W.T*(dE/dY)(output_gradient)
        return np.dot(output_gradient, self.weights.T)
    
class Activation(Layer):
    def __init__(self):
        pass

    def feed_forward(self, input):
        self.input = input
        return sigmoid(self.input)
    
    def backpropagation(self, output_gradient, learning_rate):
        return sigmoid_prime(self.input)*output_gradient

def sigmoid(linear_pred):
    return (1 / (1 + np.exp(-linear_pred)))

def sigmoid_prime(x):
    return sigmoid(x)*(1- sigmoid(x)) 

def mse(y_true, y_pred):
    squared_error = np.square(np.subtract(y_true, y_pred))

    return squared_error.mean()

def mse_prime(y_true, y_pred):
    return(2*(y_pred-y_true)/y_true.size)

class NN:
    def __init__(self, num_of_layers, size_of_layers, epochs = 1000, learning_rate = 0.1):
        self.num_of_layers = num_of_layers
        self.size_of_layers = size_of_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.network = []

    def train(self, X, Y):
        self.network = []
        last_output = X.shape[1]
        for i in range(self.num_of_layers):
            # We initialize every layer with the input size which is equal to the output of the previous layer
            self.network.append(Layer(last_output, self.size_of_layers[i]))
            last_output = self.size_of_layers[i]
            self.network.append(Activation())

        for i in range(self.epochs):
            error = 0
            for x, y in zip(X, Y):
                ### Feed Forward ###
                output = x
                output = output.reshape((1, x.shape[0]))
                
                for layer in self.network:
                    output = layer.feed_forward(output)
                    if output.shape[0] != 1: 
                        output = np.transpose(output)

                # Calculate the error for each sample
                error += mse(y, output)

                ### Backpropagation ###
                output_gradient = mse_prime(y, output)
                for layer in reversed(self.network):
                    output_gradient = layer.backpropagation(output_gradient, self.learning_rate)

            error /= X.shape[0]

            print("{}/{} loss: {} ".format(i+1, self.epochs, error))

    def predict(self, X):
        output_result = []
        for x in X:
            output = x
            output = output.reshape((1, x.shape[0]))
            for layer in self.network:
                output = layer.feed_forward(output)
                if output.shape[0] != 1: 
                    output = np.transpose(output)
            
            output_result.append(output)

        return output_result