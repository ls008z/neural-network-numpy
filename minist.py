import numpy as np
import neuralnet as nn
from numpy import genfromtxt

train_raw = genfromtxt('mnist_train.csv', delimiter=',')
test_raw = genfromtxt('mnist_test.csv', delimiter=',')

x_train = train_raw[:, 1:].reshape(train_raw.shape[0], 1, 28*28)
x_train.astype('float32')
x_train /= 255

x_test = test_raw[:, 1:].reshape(test_raw.shape[0], 1, 28*28)
x_test.astype('float32')
x_test /= 255


def categorize_output(y):
    rows = []
    for i in range(len(y)):
        row = [1*(j == y[i]) for j in range(10)]
        rows.append(row)
    output = np.array(rows).reshape(len(rows), 1, 10)
    output.astype('float32')
    return output


y_train = categorize_output(train_raw[:, 0])
y_test = categorize_output(test_raw[:, 0])


net = nn.Network()
net.add_layer(nn.FullyConnectedLayer(28*28, 100))
net.add_layer(nn.ActivationLayer(nn.tanh, nn.tanh_gradient))
net.add_layer(nn.FullyConnectedLayer(100, 50))
net.add_layer(nn.ActivationLayer(nn.tanh, nn.tanh_gradient))
net.add_layer(nn.FullyConnectedLayer(50, 10))
net.add_layer(nn.ActivationLayer(nn.tanh, nn.tanh_gradient))


train_size = 60000
test_size = 10000

net.set_loss(nn.mse, nn.mse_gradient)

net.fit(x_train=x_train[0:train_size], y_train=y_train[0:train_size],
        epochs=35, learning_rate=0.1)

predict = net.predict(x_test[0:test_size])
predict = [predict[i].argmax() for i in range(test_size)]

true_value = list(test_raw[0:test_size, 0].astype('int'))

correct = [predict[i] == true_value[i] for i in range(test_size)]
error_rate = 1 - sum(correct)/len(correct)

print('\n')
# print(predict)
# print(true_value)
print(error_rate)
