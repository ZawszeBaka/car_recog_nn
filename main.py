import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import cv2
import pickle
from pprint import pprint

def sigmoid(z):
    '''
    Computes the sigmoid of x for each x in lst_vals
                    1
    Sigmoid(x) = ---------
                      -x
                  1 + e
    Args:
    lst_vals -- input values, list

    Returns:
    '''
    x = tf.placeholder(tf.float32,name='x')

    # Define sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a sess, run it ,
    with tf.Session() as sess:
        print('[INFO] x = ', sess.run(x, feed_dict={x:z}), end='')
        rs = sess.run(sigmoid, feed_dict={x:z})
        print(' , sigmoid(x) = ', rs)

    return rs

def compute_cost_v1(logits, labels):
    '''
    Computes the cost using the sigmoid cross entropy

    Args:
    logits -- vector containing z, output of the last linear unit
            (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: logits will feed into z, labels into y
    '''
    # placeholders for logits z and labels y
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    # Use the loss function
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)

    with tf.Session() as sess :
        cost = sess.run(cost, feed_dict={z:logits, y:labels})

    return cost

def one_hot_matrix(labels, C):
    '''
    Creates a matrix where the i-th row corresponds to the ith class
    number and the jth column corresponds to the jth training example
    So if example j had a label i. then entry (i,j) will be 1

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    '''

    # Use tf.one_hot , be careful with the axis
    one_hot_matrix = tf.one_hot(labels, C, axis=1)

    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)

    return one_hot

def print_some_pics(X_train_orig, Y_train_orig, num_pics = 10):
    for i in range(num_pics):
        print('[INFO] Label: ', Y_train_orig[0,i])
        cv2.imshow('Sign',X_train_orig[i])
        cv2.waitKey()
    cv2.destroyAllWindows()

def preprocessing():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # print_some_pics(X_train_orig, Y_train_orig)
    print('[INFO] Classes ', classes)
    print('[INFO] Train info ', X_train_orig.shape)
    print('[INFO] Test info ', X_test_orig.shape)

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Normalize image vectors
    X_train = X_train_flatten/255
    X_test = X_test_flatten/255

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, len(classes))
    Y_test = convert_to_one_hot(Y_test_orig, len(classes))

    # Note that 12288 = 64x64x3
    print('[INFO] Train info after normalizing :', X_train.shape)

    return X_train, Y_train, X_test, Y_test

def create_placeholders(n_x, n_y):
    '''
    Creates the placeholders for the tensorflow session

    Arguments:
    n_x -- scalar, size of an image vector ( = 64x64x3 )
    n_y -- scalar, number of classes (from 0 to 5, so ->6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype float
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype float

    Tips:
    You will use None because it let's us be flexible on the number of examples
    you will for the placeholders
    In fact, the number of examples during test/train is different
    '''
    X = tf.placeholder(tf.float32, shape=(n_x,None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y,None), name='Y')
    print('[INFO] Creating placeholders X, Y ', X, Y )
    return X,Y

def initialize_parameters(flat_shape,uni_labels):
    '''
    Arguments:
        flat_shape: 68x68x3 = 12288
        uni_labels: num classes

    initialize parameters to build a neural network with tensorflow
    The shapes are
        W1: [25,12288]
        b1: [25,1]
        W2: [12,25]
        b2: [12,1]
        W3: [2,12]
        b3: [2,1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    '''
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', [25,4096], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12,25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [2,12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [2,1], initializer = tf.zeros_initializer())

    print('[INFO] W1 ', W1)
    print('[INFO] b1 ', b1)
    print('[INFO] W2 ', W2)
    print('[INFO] b2 ', b2)
    print('[INFO] W3 ', W3)
    print('[INFO] b3 ', b3)
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2,
                  'W3': W3,
                  'b3': b3}

    return parameters

def forward_propagation(X, parameters):
    '''
    Implements the forward propagation for the model
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Args:
    X -- input dataset placeholder, of shape (input_size, number_of_examples)
    parameters -- python dictionary containing your parameters 'W1', 'b1',...'W3', 'b3'

    Returns:
    Z3 -- the output of the last LINEAR unit
    '''

    # Retrieve the parameters from the dictionary 'parameters'
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    #
    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)
    print('[INFO] Z3 = ', Z3)

    return Z3

def compute_cost(Z3, Y):
    '''
    Compute the cost

    Args:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape
          (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    '''

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, mini_batch_size = 32, print_cost = True):
    '''
    Implements a three-layer tensorflow neural network:
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Args:
    X_train -- training set, of shape (64x64x3, number of training examples = 1080)
    Y_train -- label, of shape (distinct labels = 6 , num of training examples = 1080)
    X_test -- training set, of shape (64x64x3, number of test examples = 120)
    Y_test -- label, of shape (distinct labels = 6 , num of test examples = 120)
    learning_rate -- learning_rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    mini_batch_size -- size of a mini batch
    print_cost -- if True, print the cost every 100 epochs

    Returns :
    parameters -- parameters learnt by the model. they can then be used to predict
    '''
    # to be able to rerun the model without overwriting tf vars
    ops.reset_default_graph()
    # to keep consistent results
    tf.set_random_seed(1)
    seed = 3
    # n_x : input size , m: number of examples in the train set
    (n_x, m) = X_train.shape
    # n_y : output size
    n_y = Y_train.shape[0]
    # To keep track of the cost
    costs = []

    ## Placeholders
    # X -- placeholder for the data input, of shape [n_x, None] and dtype float
    # Y -- placeholder for the input labels, of shape [n_y, None] and dtype float
    X, Y = create_placeholders(n_x,n_y)

    ## Initialize parameters: W1,b1, W2,b2, W3,b3
    # W1: [25,4096]
    # b1: [25,1]
    # W2: [12,25]
    # b2: [12,1]
    # W3: [2,12]
    # b3: [2,1]
    uni_labels = np.unique(Y_train).shape[0]
    def count_pixels(X):
        m = 1
        for i in X.shape[1:]:
            m *= i
        return m
    parameters = initialize_parameters(count_pixels(X_train), uni_labels)

    # Forward propagation: Z1,A1, Z2,A2, Z3
    # Z1 = tf.add(tf.matmul(W1,X), b1)
    # A1 = tf.nn.relu(Z1)
    # Z2 = tf.add(tf.matmul(W2,A1), b2)
    # A2 = tf.nn.relu(Z2)
    # Z3 = tf.add(tf.matmul(W3,A2), b3)
    Z3 = forward_propagation(X,parameters)

    # Cost function
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    cost = compute_cost(Z3, Y)

    # Back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_mini_batches = int(m/mini_batch_size)
            seed = seed + 1
            mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)

            for mini_batch in mini_batches:
                # Select a mini_batch
                (mini_batch_X, mini_batch_Y) = mini_batch

                # The line that runs the graph on a minibatch
                # Run the session to execute the "optimizer" and the "cost"
                # the feed_dict should contain a minibatch for X, Y
                _, mini_batch_cost = sess.run([optimizer,cost],
                                              feed_dict={X:mini_batch_X,
                                                         Y:mini_batch_Y})

                epoch_cost += mini_batch_cost / num_mini_batches

            # Print the cost every epoch
            if print_cost and epoch % 100 == 0 :
                print('[INFO] Cost after epoch %i : %f ' % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0 :
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 5)')
        plt.title('Learning rate = '+str(learning_rate))
        plt.show()

        # Let's save the parameters in a variable
        parameters = sess.run(parameters)
        print('[INFO] Parameters have been trained ! ')

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print('[INFO] Train Accuracy: ', accuracy.eval({X:X_train, Y:Y_train}))
        print('[INFO] Test Accuracy: ', accuracy.eval({X:X_test, Y:Y_test}))

        return costs, parameters

def save_parameters(file_name,parameters):
    with open('model'+'/'+file_name,'wb') as f:
        pickle.dump(parameters, f)

def load_parameters(file_name):
    with open('model'+'/'+file_name,'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    # virtualenv --system-site-packages -p python3 ./venv
    # venv is just a name
    # source venv/bin/activate
    #  pip3 install --upgrade tensorflow

    # Training !
    X_train, Y_train, X_test, Y_test = preprocessing()

    costs, parameters = model(X_train,Y_train, X_test, Y_test,num_epochs = 1000)
    print('[INFO] Parameters')
    pprint(parameters)

    save_parameters('car_model.pickle', [costs, parameters])
    print('[INFO] Save model to file model/car_model.pickle. Done!')
