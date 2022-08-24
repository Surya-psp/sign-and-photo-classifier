import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
# import pandas as pd
import copy
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import PIL
import random

test_folder = r'D:\\SoftwareEngineering\\Extracted Faces\\Extracted Faces'
plt.figure(figsize=(20,20))
files = os.listdir(test_folder)
for i in range(10):
    files = random.choice(os.listdir(test_folder))
#     print(files)
    sub_path = os.path.join(test_folder, files)
    file = random.choice(os.listdir(sub_path))
    img_path = os.path.join(sub_path, file)
    # print(file)
    img = mpimg.imread(img_path)
    # print(img.shape)
    ax = plt.subplot(1, 10, i+1)
#     ax.title.set_text(file)
    plt.imshow(img)



def create_dataset(img_folder):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    img_data_array = []
    class_name = []
    icount, scount = 0, 0
    for dir1 in os.listdir(img_folder):
        k = os.listdir(os.path.join(img_folder, dir1))
        for i in range(len(k)):
            if icount > 1000:
                icount = 0
                break
            if scount > 1000:
                break
            image_path = os.path.join(img_folder, dir1, k[i])
            if dir1 == 'train':
#                 scount += 1
                sub_path = os.path.join(img_folder, dir1, k[i])
                for img_file in os.listdir(os.path.join(img_folder, dir1, k[i])):
                    scount += 1
                    image_path = os.path.join(img_folder, dir1, k[i], img_file)
                    img = PIL.Image.open(r""+str(image_path))
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    img = np.array(img)
                    img.astype('float32')
        #             print(img)
                    img=img/255
                    img_data_array.append(img)
                    class_name.append(dir1)
            else:
                icount += 1
            img = PIL.Image.open(r""+str(image_path))
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img)
            img.astype('float32')
#             print(img)
            img=img/255
            img_data_array.append(img)
            class_name.append(dir1)
#             count += 1
    return img_data_array, class_name

IMG_FOLDER = r'D:\\SoftwareEngineering\\BinaryTraining'

img_data, class_name = create_dataset(IMG_FOLDER)

img_data_final = []
for i in img_data:
#     print(i.shape)
    i = i.reshape(i.shape[0]*i.shape[1]*i.shape[2], 1)
    img_data_final.append(i)


coupled_dataset = list(zip(class_name, img_data_final))
random.shuffle(coupled_dataset)

class_name, img_data_final = zip(*coupled_dataset)
# print(class_name)

class_name_final = []
for i in class_name:
    if i == 'Faces':
        class_name_final.append(0)
    else:
        class_name_final.append(1)
# print(class_name_final)

X_train = np.array(img_data_final[:1900])
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
# print(len(img_data_final))
print(X_train.shape)
Y_train = np.array(class_name_final[:1900]).reshape(1900,1)
Y_train.reshape([X_train.shape[0], 1])
print(Y_train.shape)
X_test = np.array(img_data_final[1901:2001])
Y_test = np.array(class_name_final[1901:2001])


# X_train.shape
X_train = X_train.reshape((49152, 1900))
# print(X_test.shape)
X_test = X_test.reshape((49152, 100))
Y_test = Y_test.reshape((1, 100))
Y_train = Y_train.reshape((1, 1900))
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))




print(tf.__version__)


def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- Y = WX + b 
    """

    np.random.seed(1)
    
    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # (approx. 4 lines)
    # X = ...
    # W = ...
    # b = ...
    # Y = ...
    # YOUR CODE STARTS HERE
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X),b)
    
    # YOUR CODE ENDS HERE
    return Y




def sigmoid(z):
    
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- (tf.float32) the sigmoid of z
    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.
    
    # (approx. 2 lines)
    # z = ...
    # a = ...
    # YOUR CODE STARTS HERE
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    # YOUR CODE ENDS HERE
    return a






def one_hot_matrix(label, depth=6):
    """
Computes the one hot encoding for a single label
Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take

Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    # (approx. 1 line)
    # one_hot = ...
    # YOUR CODE STARTS HERE
    one_hot = tf.one_hot(label,depth,axis=0)
    one_hot = tf.reshape(one_hot, [depth,])
    # YOUR CODE ENDS HERE
    return one_hot





def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
                                
    initializer = tf.keras.initializers.GlorotNormal(seed=1)   
    #(approx. 6 lines of code)
    # W1 = ...
    # b1 = ...
    # W2 = ...
    # b2 = ...
    # W3 = ...
    # b3 = ...
    # YOUR CODE STARTS HERE
    W1 = tf.Variable(initializer(shape=([25, 12288])))
    b1 = tf.Variable(initializer(shape=([25, 1])))
    W2 = tf.Variable(initializer(shape=([12, 25])))
    b2 = tf.Variable(initializer(shape=([12, 1])))
    W3 = tf.Variable(initializer(shape=([6, 12])))
    b3 = tf.Variable(initializer(shape=([6, 1])))
    # YOUR CODE ENDS HERE
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters



def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    #(approx. 5 lines)                   # Numpy Equivalents:
    # Z1 = ...                           # Z1 = np.dot(W1, X) + b1
    # A1 = ...                           # A1 = relu(Z1)
    # Z2 = ...                           # Z2 = np.dot(W2, A1) + b2
    # A2 = ...                           # A2 = relu(Z2)
    # Z3 = ...                           # Z3 = np.dot(W3, A2) + b3
    # YOUR CODE STARTS HERE
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)
    # YOUR CODE ENDS HERE
    
    return Z3


# GRADED FUNCTION: compute_cost 

def compute_cost(logits, labels):
    """
    Computes the cost
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    #(1 line of code)
    # cost = ...
    # YOUR CODE STARTS HERE
#     tf.keras.losses.CategoricalCrossentropy(
#         from_logits=False,
#         label_smoothing=0.0,
#         axis=-1,
#         reduction=tf.keras.losses_utils.ReductionV2.AUTO,
#         name='categorical_crossentropy'
#     )
#     cost = tf.keras.losses.CategoricalCrossentropy()
#     cost(labels, logits).numpy()
#     print(cost)
    k = tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits))
#     print(k)
#     print(tf.keras.losses.categorical_crossentropy(logits, labels))
#     print((k[0] + k[1])/2)
    cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits)))
#     YOUR CODE ENDS HER
    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    costs = []                                        # To keep track of the cost
    train_acc = []
    test_acc = []
    
    # Initialize your parameters
    #(1 line)
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # The CategoricalAccuracy will track the accuracy for this multiclass problem
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()
    
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
    #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster 

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.
        
        #We need to reset object to start measuring from 0 the accuracy each epoch
        train_accuracy.reset_states()
        
        for (minibatch_X, minibatch_Y) in minibatches:
            
            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                # 2. loss
                minibatch_cost = compute_cost(Z3, tf.transpose(minibatch_Y))

            # We accumulate the accuracy of all the batches
            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost
        
        # We divide the epoch cost over the number of samples
        epoch_cost /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())
            
            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()


    return parameters, costs, train_acc, test_acc




x_train = tf.data.Dataset.from_tensor_slices(train_dataset['X_train'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['Y_train'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['X_test'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['Y_test'])




parameters, costs, train_acc, test_acc = model(x_train, y_train, x_test, y_test, num_epochs=100)