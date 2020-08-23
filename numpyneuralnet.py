import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def ReLu(x):
    return np.maximum(0, x)


N = 100  # number of points per class
n_inputs = 2  # dimensionality
n_classes = 3  # number of classes
input_data = np.zeros((N*n_classes, n_inputs))  # data matrix (each row = single example)
labels = np.zeros(N*n_classes, dtype='uint8')  # class labels
for j in range(n_classes):
  ix = range(N*j, N*(j+1))
  r = np.linspace(0.0, 1, N)  # radius
  t = np.linspace(j*4, (j+1) * 4, N) + np.random.randn(N)*0.2  # theta
  input_data[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  labels[ix] = j
plt.scatter(input_data[:, 0], input_data[:, 1], c=labels, s=40, cmap=plt.cm.Spectral)
plt.show()

# initialize parameters randomly
h = 100  # size of hidden layer
W1 = np.random.randn(n_inputs, h)
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, n_classes)
b2 = np.zeros((1, n_classes))

LR = 0.1

# gradient descent loop
num_examples = input_data.shape[0]

for i in range(10000):

    hidden_layer = ReLu(np.matmul(input_data, W1) + b1)
    scores = np.matmul(hidden_layer, W2) + b2
    predictions = softmax(scores)

    corect_logprobs = -np.log(predictions[range(num_examples), labels])
    loss = np.sum(corect_logprobs) / num_examples
    if i % 10 == 0:
        print("iteration %r: loss %f" % (i, loss))

    dscores = predictions
    dscores[range(num_examples), labels] -= 1
    dscores /= num_examples

    # f = w*x + b

    dW2 = np.matmul(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dhidden = np.matmul(dscores, W2.T)

    dhidden[hidden_layer <= 0] = 0

    dW = np.matmul(input_data.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    W1 += -LR * dW
    b1 += -LR * db
    W2 += -LR * dW2
    b2 += -LR * db2













