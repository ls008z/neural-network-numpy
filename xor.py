import numpy as np
import neuralnet as nn

x_train = np.array([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]]
])

y_train = np.array([
    [[0]],
    [[1]],
    [[1]],
    [[0]]
])

net = nn.Network()
net.add_layer(nn.FullyConnectedLayer(2, 3))
net.add_layer(nn.ActivationLayer(nn.tanh, nn.tanh_gradient))
net.add_layer(nn.FullyConnectedLayer(3, 1))
# think of it as a binary prediction problem, so we add the last activation layer too
net.add_layer(nn.ActivationLayer(nn.tanh, nn.tanh_gradient))


net.set_loss(nn.mse, nn.mse_gradient)
net.fit(x_train=x_train, y_train=y_train, epochs=1000, learning_rate=0.1)

out = net.predict(x_train)
print(out)
