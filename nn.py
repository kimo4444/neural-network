import numpy as np
# training set
inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]])
outputs = np.array([[0], [1], [1], [0]])


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # assigning random weigths
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))

# derivative of sigmoid for gradient descent

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self):
        self.layer1 = self.sigmoid_function(np.dot(self.input, self.weights1))
        self.output = self.sigmoid_function(np.dot(self.layer1, self.weights2))

    def backpropagate(self):
        d_weights2 = np.dot(
            self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(
            self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, number_of_iterations):
        for iteration in range(number_of_iterations):
            self.feedforward()
            self.backpropagate()


if __name__ == "__main__":

    neural_network = NeuralNetwork(inputs, outputs)
    neural_network.train(2000)

    print(neural_network.output)
