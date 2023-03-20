import numpy as np


class Layer:
    """Base Layer Class
    A base class that lays out the structure of layers. Not really
    used in anyway.
    """

    def __init__(self):
        """
        A layer takes a vector of inputs X and produce a vector of output Y.
        Here, both are row vectors, namely (1 by n) numpy arrays
        """
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        Computes the output Y of a layer for a given input X.
        """
        raise NotImplementedError

    def backward_propagation(self, output_gradient, learning_rate):
        """
        Backward propagation is the essense of neural network.
        Compute input gradient dE/dX from output gradient dE/dY.
        Update parameters using dE/dW gradient decent.
        The so-called gradients are all (1 by n) row vectors. The values
        are partial derivatives evaluated at the current value.
        (Definitely abusing notations here. The input X of a layer
        is the output Y from the previous layer.)
        """
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    """
    A layer that linearly (affine) trasforms input X
    to output Y
    """

    def __init__(self, input_size, output_size):
        """These layers are instantiated when we add them to the
        Network object. When we do so, we have to specify the size 
        of input and output, i.e. the width of the network at this
        layer. 

        When a layer is instantiated, it also initializes a weight 
        matrix W and a bias vector B.

        Args:
            input_size (int): length of input vector
            output_size (int): length of output vector
        """
        super().__init__()
        # (len(X), len(Y)) shape ndarray
        self.weights = np.random.rand(input_size, output_size) - 0.5

        # (1, len(Y)) shape ndarray
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input):
        """ Calculate output Y from input X
        Y = X W + B
        Y, X and B are row vectors

        Args:
            input (ndarray): (1, len(X)) ndarray

        Returns:
            output (ndarray): (1, len(Y)) ndarray 
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        """applies chain rule to calculate gradients
        pass dE/dX as output to the previous layer
        applies gradient descent

        W <- W - a * dE/dW

        Args:
            output_gradient (ndarray): dE/dY, (1, len(Y)) ndarray,
                values of the partial derivatives evaluated at current Y.
            learning_rate (float): a in  W <- W - a * dE/dW

        Returns:
            input_gradient (ndarray): (1, len(X)) ndarray,
                values of the partial derivatives evaluated at current X.
        """
        # so simple because it's linear
        # note that self.weights and self.input are all values
        # i.e. the derivatives here are "evaluated at current point"
        input_gradient = np.dot(output_gradient, self.weights.T)
        # note here how input is directly read as class attribute
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = output_gradient

        # gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient


class ActivationLayer(Layer):
    """
    It's so cool to treat activation as another layer.
    The nonlinear gradient is handled very elegantly this way.
    """

    def __init__(self, activation, activation_gradient):
        """pass the output through activation function
        also apply chain rules when back propagation

        Args:
            activation (function): a R -> R function
            activation_gradient (function): a R -> R function
        """
        super().__init__()
        self.activation = activation
        self.activation_gradient = activation_gradient

    def forward_propagation(self, input):
        """element-wise apply the activation function

        Args:
            input (ndarray): (1, len(Y)) ndarray

        Returns:
            output (ndarray): (1, len(Y)) ndarray
        """
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        """apply chain rule

        Args:
            output_gradient (ndarray): dE/dY, (1, len(Y)) ndarray
            learning_rate (None): don't need this

        Returns:
            input_gradient: (1, len(Y)) ndarray
        """
        # note that every thing is calculated element-wise
        # note that self.input is plugged into the activation_gradient function
        # i.e. we are evaluating the partial derivative at the current value of Y
        return self.activation_gradient(self.input) * output_gradient
