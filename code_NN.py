import tensorflow as tf

from keras.src.metrics.accuracy_metrics import accuracy

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
import numpy as np

x_train = x_train/255.0
x_test = x_test/255.0

X = (x_train[0]).flatten()

W1 = np.random.randn(784,128)*0.01
W2 = np.random.randn(128,10)*0.01
B1 = np.zeros((128,))
B2 = np.zeros((10,))

def relu(z):
    return np.maximum(0,z)

def softmax(scores):
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    probabilities = exps / np.sum(exps)
    predicted_class = np.argmax(probabilities)
    return probabilities, predicted_class

def compute_cross_entropy(probabilities, true_label):
    return -np.log(probabilities[true_label])

# Forward pass
Z1 = X @ W1 + B1      # layer 1 pre-activation
A1 = relu(Z1)         # layer 1 post-activation

Z2 = A1 @ W2 + B2     # layer 2 pre-activation
A2, predicted_class = softmax(Z2)   # layer 2 post-activation (probabilities)

loss = compute_cross_entropy(A2, y_train[0])
print("Predicted class:", predicted_class)
print("True label:", y_train[0])
print("Loss:", loss)


y_onehot = np.zeros(10)
y_onehot[y_train[0]] = 1
dA2 = np.zeros(10)
dA2[y_train[0]]=-1/A2[y_train[0]]
print("dA2:", dA2,"\n",dA2.shape)
dZ2 = A2 - y_onehot
print("dZ2:", dZ2,"\n",dZ2.shape)
dW2 = np.outer(A1,dZ2)
print("dW2:", dW2,"\n",dW2.shape)
dB2 = dZ2
print("dB2:", dB2,"\n",dB2.shape)
dA1 = dZ2 @ W2.T
print("dA1",dA1,"\n",dA1.shape)
dZ1 = dA1 * (Z1>0)
print("dZ1",dZ1,"\n",dZ1.shape)
dB1 = dZ1
print("dB1",dB1,"\n",dB1.shape)
dW1 = np.outer(X,dZ1)
print("dW1",dW1,"\n",dW1.shape)


learning_rate = 0.1

W2 = W2 - learning_rate * dW2
B2 = B2 - learning_rate * dB2
W1 = W1 - learning_rate * dW1
B1 = B1 - learning_rate * dB1




Z1 = X @ W1 + B1      # layer 1 pre-activation
A1 = relu(Z1)         # layer 1 post-activation

Z2 = A1 @ W2 + B2     # layer 2 pre-activation
A2, predicted_class = softmax(Z2)   # layer 2 post-activation (probabilities)

loss = compute_cross_entropy(A2, y_train[0])
print("Predicted class:", predicted_class)
print("True label:", y_train[0])
print("Loss:", loss)


### Batch forward pass
x_batch = x_train[0:32].reshape(32,784)
y_batch = y_train[0:32]
N = len(y_batch)
print(x_train.shape)

Z1 = x_batch @ W1 + B1
print(Z1.shape)
A1 = relu(Z1)
print(A1.shape)


Z2 = A1 @ W2 + B2
print(Z2.shape)
def softmax_batch(scores):
    shifted = scores - np.max(scores,axis=1,keepdims=True)
    exps = np.exp(shifted)
    return exps/np.sum(exps,axis=1,keepdims=True)
A2 = softmax_batch(Z2)
print(A2,A2.shape)


def calc_loss_batch(scores,classes):
    correct_classes_probs = scores[np.arange(scores.shape[0]),classes]
    loss_batch = -np.log(correct_classes_probs)
    return np.mean(loss_batch),correct_classes_probs
print(calc_loss_batch(A2,y_batch))


y_onehot_batches = np.zeros((32,10))
y_onehot_batches[np.arange(32),y_batch] = 1
dZ2 = A2 - y_onehot_batches

dW2 = A1.T @ dZ2 / N
print(dW2.shape)

dB2 = dZ2.sum(axis = 0)/N
print(dB2.shape)

dA1 = dZ2 @ W2.T
print(dA1.shape,dA1)

dZ1 = dA1 * (Z1 > 0)
print(dZ1,dZ1.shape)

dB1 = dZ1.sum(axis = 0)/N
print(dB1.shape)

dW1 = x_batch.T @ dZ1/N
print(dW1.shape)



W2 = W2 - learning_rate * dW2
B2 = B2 - learning_rate * dB2
W1 = W1 - learning_rate * dW1
B1 = B1 - learning_rate * dB1


def train_step(x_batch, y_batch , W1 , W2 , B1 , B2 , learning_rate ):
    N = len(y_batch)
    ## forward pass:

    Z1 = x_batch @ W1 + B1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + B2
    A2 = softmax_batch(Z2)

    loss , class_probs = calc_loss_batch(A2 , y_batch)

    ## backward pass

    y_onehot_batches2 = np.zeros((N,10))
    y_onehot_batches2[np.arange(N),y_batch] = 1


    dZ2 = A2 - y_onehot_batches2
    dW2 = A1.T @ dZ2 / N
    dB2 = dZ2.sum(axis = 0) / N
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1>0)
    dW1 = x_batch.T @ dZ1 / N
    dB1 = np.sum(dZ1 , axis = 0) / N

    ## gradient descent


    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1



    return W2,B2,W1,B1,loss

epochs = 175

learning_rate = 0.1
x_train_flattened = x_train.reshape(60000,784)
num_batches = len(y_train)//32
def train_loop(num_batches , x_train_flattened, y_train , W1 , W2 , B1 , B2):
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            start = 32*i
            end = 32*(i+1)
            W2 , B2 , W1 , B1 , loss = train_step(x_train_flattened[start:end] , y_train[start:end] , W1 , W2 , B1 , B2 , learning_rate)
            epoch_loss += loss
        print(f"epoch{epoch+1} loss:",epoch_loss,"\n",f"average epoch{epoch+1} loss:{epoch_loss/num_batches}")
    return W2,B2,W1,B1
W2, B2, W1, B1 = train_loop(num_batches , x_train_flattened, y_train , W1 , W2 , B1 , B2)


def validate_on_test_data(W1,B1,W2,B2,x,y):
    Z1 = x @ W1 + B1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + B2
    A2 = softmax_batch(Z2)
    predictions = np.argmax(A2,axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy
x_test_flattened = x_test.reshape(10000,784)
accuracy = validate_on_test_data(W1,B1,W2,B2,x_test_flattened,y_test)
print(accuracy)
