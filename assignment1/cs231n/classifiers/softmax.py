import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    c = - np.max(scores)
    exp_sum = 0.0

    for j in xrange(num_classes):
      exp_sum += np.exp(scores[j] + c)

    for j in xrange(num_classes):
      if y[i] == j:
        dW[:,j] -= X[i]
      dW[:,j] += np.exp(scores[j] + c)/exp_sum * X[i]

    loss += np.log(exp_sum)
    loss -= scores[y[i]] + c
  
  # Normalize loss and gradient
  loss /= num_train
  dW /= num_train

  # Add regularization terms to loss and gradient
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  c = - np.max(scores, axis=1)[:,np.newaxis]
  y_hat = np.exp(scores + c) / np.sum(np.exp(scores + c), axis=1)[:,np.newaxis]
  y_true = np.zeros((num_train, num_classes))
  y_true[range(num_train),y] = 1

  loss -= np.sum(y_true * np.log(y_hat))
  dW += X.T.dot(y_hat - y_true)
  
  # Normalize loss and gradient
  loss /= num_train
  dW /= num_train

  # Add regularization terms to loss and gradient
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

