class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_gradient = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_gradient):
        """set loss functions

        Args:
            loss (function): a function that calcuates loss
            loss_gradient (function): a function that evaluates dE/dY
        """
        self.loss = loss
        self.loss_gradient = loss_gradient

    def predict(self, input_data):
        samples_size = len(input_data)
        result = []

        # a very cool loop
        # starts from input data
        # uses the forward_propagation method in each layer
        # output is passed into the same method from next layer
        for i in range(samples_size):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        """Can you belive the whole fit is 16 lines of code?

        Args:
            x_train (ndarray): (n, 1, p) ndarray, so each 
                observation is (1, p) ndarray
            y_train (ndarray): (n, 1, q) ndarray, so each
                label is (1, q) ndarray
            epochs (int): number of iterations
            learning_rate (float): learning rate
        """
        sample_size = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(sample_size):
                # the cool loop again
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # have to sum up errors for every observation
                err += self.loss(y_train[j], output)

                # a even cooler loop
                # dE/dY is first evaluated at the the output using
                # the analytical loss_gradient function
                gradient = self.loss_gradient(y_train[j], output)
                # now the gradient is passed layer by layer
                # backward, recall that backward_propagation method
                # of each layer returns a gradient and also
                # updates the parameters as a side effect,
                # which are stored as attributes to those layer classes
                # great demonstration of objective oriented programming
                for layer in reversed(self.layers):
                    gradient = layer.backward_propagation(
                        gradient, learning_rate)
            # the error is really just for us to take a look
            # it is not used in the algorithm
            err /= sample_size
            print(f'epoch {i+1}/{epochs}   error={err}')
