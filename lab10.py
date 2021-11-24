import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, net_arch):
        """
                this function initialize the weights with random values (-1,1)
                Parameters
                ----------
                self: the class object itself

                net_arch:  consists of a list of integers, indicating
                           the number of neurons in each layer, i.e. the network architecture

        """
        self.layers = len(net_arch)  # 3 layers: the 1 is input layer, ...
        self.arch = net_arch
        self.weights = []  # list of arrays for the weights

        # Random initialization with range of weight values (-1,1)
        for layer in range(self.layers - 1):
            w = 2 * numpy.random.rand(net_arch[layer] + 1, net_arch[layer + 1]) - 1  # 1st array - 3x2, 2d - 3x1
            self.weights.append(w)
        # Note that a bias unit is added to each hidden layer and a “1” will be added to the input layer. That’s why
        # the dimension of weight matrix is (nj+1)×nj+1 instead of nj×nj+1.

    def _forward_prop(self, x):
        """
                Forward propagation propagates the sampled input data forward
                through the network to generate the output value.
                Parameters
                ----------
                self: the class object itself

                x:  the sampled input data

        """
        for i in range(len(self.weights) - 1):  # in our case 1 time
            activation = numpy.dot(x[i], self.weights[i])  # Multiply the input data by weights to form a hidden layer
            activity = sigmoid(activation)

            # add the bias for the next layer
            activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
            x.append(activity)

        # last layer
        activation = numpy.dot(x[-1], self.weights[-1])
        activity = sigmoid(activation)
        x.append(activity)

        return x

    def loss(self, y, target):
        """
                generates the deltas (the difference between the targeted and actual output values)
                of all output and hidden neurons
                Parameters
                ----------
                self: the class object itself
                y:  actual output values calculated at the stage of forward prop
                target: targeted values
        """
        error = target - y[-1]
        delta_vec = [error * sigmoid_derivative(y[-1])]

        # we need to begin from the back, from the next to last layer
        for i in range(self.layers - 2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error * sigmoid_derivative(y[i][1:])
            delta_vec.append(error)

        # Now we need to set the values from back to front
        delta_vec.reverse()

        return delta_vec

    def _back_prop(self, y, delta_vec, learning_rate):
        """
                adjust the weights using gradient descent
                Parameters
                ----------
                self: the class object itself
                y:  actual output values calculated at the stage of forward prop
                delta_vec: the gradient of weights
                learning_rate: learning rate
        """

        # Finally, we adjust the weights, using the backpropagation rules
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.arch[i] + 1)
            delta = delta_vec[i].reshape(1, self.arch[i + 1])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    def fit(self, data, labels, learning_rate=0.1, epochs=100):
        """
                adjust the weights using gradient descent
                Parameters
                ----------
                self: the class object itself
                data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
                labels:  the result of the logical operation 'xor' on each of those input pairs
                learning_rate: learning rate
                epochs: number of epochs
        """

        # Add bias units to the input layer -
        # add a "1" to the input data (the always-on bias neuron)
        ones = numpy.ones((1, data.shape[0]))  # [[1. 1. 1. 1.]]
        Z = numpy.concatenate((ones.T, data), axis=1)  # [[1. 0. 0.]
                                                       #  [1. 0. 1.]
                                                       #  [1. 1. 0.]
                                                       #  [1. 1. 1.]]
        for k in range(epochs):
            if (k + 1) % 10000 == 0:
                print('epochs: {}'.format(k + 1))

            sample = numpy.random.randint(X.shape[0])  # random number from [0, 4)

            # We will now go ahead and set up our feed-forward propagation:
            x = [Z[sample]]  # a row with index sample from Z, for ex. [array([1., 1., 1.])]
            y = self._forward_prop(x)

            # Now we do our back-propagation of the error to adjust the weights:
            target = labels[sample]
            delta_vec = self.loss(y, target)
            self._back_prop(y, delta_vec, learning_rate)

    def predict_single_data(self, x):
        """
                is used to check the prediction result of this neural network.
                Parameters
                ----------
                self: the class object itself
                x: single input data
        """
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = sigmoid(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return val[1]

    def predict(self, X):
        """
                is used to check the prediction result of this neural network.
                Parameters
                ----------
                self: the class object itself
                X: the input data array
        """
        Y = numpy.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            Y = numpy.vstack((Y, y))
        return Y


if __name__ == "__main__":

    nn = NeuralNetwork([2, 2, 1])

    # Set the input data
    X = numpy.array([[0, 0], [0, 1],
                     [1, 0], [1, 1]])

    # Set the labels, the correct results for the xor operation
    y = numpy.array([0, 1, 1, 0])

    # Call the fit function and train the network for a chosen number of epochs
    nn.fit(X, y, epochs=100000)

    # Show the prediction results
    y_pred = nn.predict(X)
    for a, b, c in zip(X, y_pred, y):
        print("input:{}   prediction:{}   truth:{}".format(a, b, c))
