#!/usr/bin/env python
# encoding: utf-8
"""Simple decission trees"""

from __future__ import division, print_function
import numpy as np
from functools import partial


def itemfreq(arr, return_uvals=False):
    """Computes the number of occurences of the unqiue values of a.

    data -- array-like object

    Keyword arguments:
    return_uvals -- Return the unique values as second return value.

    Returns:
    freq -- array with number of occurences of unqiue elements
    """
    uvals, inv = np.unique(arr, return_inverse=True)
    if return_uvals:
        return uvals, np.bincount(inv)
    else:
        return np.bincount(inv)


def entropy(sample):
    """Compute the entropy H(y) = - Σ_i p_i * log_2 p_i, where p_i is the
    frequency of the i-th unique value in the sample.

    sample[:] -- Array with samples
    Returns:
    entropy of the sample
    """
    frequencies = itemfreq(sample).astype('float')
    frequencies /= np.sum(frequencies)
    return -np.sum(frequencies * np.log2(frequencies))


def information_gain(sample, sel):
    """Compute the information gain I(y|A) = H(y) - Σ_(a∈ A) p_a * H(y|a),
    where A is a binary subdivision of y into groups a=True, False and p_a is
    the frequency of the group a in A. H(y|a) is the entropy of all sample in
    the group a.

    sample -- Array
    sel -- Array of bool with same shape as sample

    Returns:

    """
    p_true = np.sum(sel) / np.size(sel)
    return entropy(sample) - p_true * entropy(sample[sel]) \
            - (1-p_true) * entropy(sample[~sel])


class ClassificationTree(object):

    """Docstring for ClassificationTreeNode. """

    def __init__(self, predictors, responses, gain_min=0.01):
        """@todo: to be defined1.

        predictor -- Predictors to fit
        response -- Labels corresponding to predictors

        """
        self._nrsamples = responses.size
        n_features = np.shape(predictors)[1]
        decision_function = lambda i, x: x[i] < np.mean(predictors[:, i])
        decfuns = [partial(decision_function, i) for i in range(n_features)]
        gains = [information_gain(responses, df(predictors.T))
                 for df in decfuns]
        decind = np.argmax(gains)

        print([information_gain(responses, df(predictors.T))
               for df in decfuns])
        print(self._nrsamples)

        self._decfun = decfuns[decind]
        self._decfun.func_doc = \
                "x[{}] < {}".format(decind, np.mean(predictors[:, decind]))
        sel = self._decfun(predictors.T)

        if gains[decind] < gain_min:
            self._tnode = itemfreq(responses[sel], return_uvals=True)
            self._fnode = itemfreq(responses[~sel], return_uvals=True)
        else:
            self._tnode = ClassificationTree(predictors[sel], responses[sel],
                                             gain_min)
            self._fnode = ClassificationTree(predictors[~sel], responses[~sel],
                                             gain_min)

    def predict(self, predictor):
        """@todo: Docstring for predict.

        predictor -- Predictors to use
        Returns:

        """
        node = self._tnode if self._decfun(predictor) else self._fnode

        try:
            return node.predict(predictor)
        except AttributeError:
            return node

    def __str__(self):
        return self._decfun.func_doc


if __name__ == '__main__':
    from sklearn.metrics import classification_report
    DATA = np.loadtxt('../data/train.csv', delimiter=',')
    X, Y = DATA[:, 1:], DATA[:, 0].astype(int)

    A = ClassificationTree(X, Y, gain_min=.1)

    TEST = np.loadtxt('../data/test.csv', delimiter=',')
    X, Y = DATA[:, 1:], DATA[:, 0].astype(int)
    pred = [A.predict(x) for x in X]
    Y_pred = [p[0][np.argmax(p[1])] for p in pred]
    print(classification_report(Y, Y_pred))
