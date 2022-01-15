
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header=None,
                           names=['x%i' % (i) for i in range(8)] + ['y'])
    # take out feature data as array
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])
    # add 1. infront of each roww
    X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


#
def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(
        feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

# the loss function used by logreg


def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss

# y_hat = 1/1+e^-(the0+the1*x+the2*x^2....)


def logreg_sgd(X, y, alpha=.001, epochs=10000, eps=1e-4):
    # TODO: compute theta(array)
    # alpha: step size
    # epochs: max epochs
    # eps: stop when the thetas between two epochs are all less than eps
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    # check if converge
    converge = False

    # epochs
    for s in range(epochs):
        # print("train: ", s)
        if (converge):
            break
        # record pre_theta
        old_theta = theta
        # take a feature index to update the theta once
        # use the whole dataset to finish a epoch
        for idx in range(n):
            xdata = X[idx, :]  # x_row
            y_hat = 1./(1+math.exp(numpy.dot(-1*xdata, theta)))
            xdata = xdata.reshape(d, 1)
            func_diff = xdata * (y[idx] - y_hat)  # j(t)微分
            func_diff_two = numpy.array(func_diff).reshape(d, 1)
            theta = theta + (alpha * func_diff_two)
        converge = True
        # TODO:
        for i in range(d):
            sub = abs(theta[i]-old_theta[i])

            if(sub < 1e-4):
                continue
            else:
                converge = False
                break
    # print(old_theta)
    return theta

# predictor


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    # tpr = True positive rate true中 判斷正確的  tp / tp+fn
    # fpr = false positive rate false中 誤判成true的 fp / tn+fp
    tpr = []
    fpr = []
    thresholds = numpy.unique(y_prob)

    for thr in thresholds:
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for index, pred in enumerate(y_prob):
            if thr > pred and y_test[index] == 1:
                fn += 1
            elif thr <= pred and y_test[index] == 1:
                tp += 1
            elif thr > pred and y_test[index] == 0:
                tn += 1
            elif thr <= pred and y_test[index] == 0:
                fp += 1
            else:
                print(thr, "   ", pred)
        tpr_tmp = .0
        fpr_tmp = .0
        tpr_tmp = tp / (tp+fn)
        fpr_tmp = fp / (tn+fp)
        tpr.append(tpr_tmp)
        fpr.append(fpr_tmp)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")


def main(argv):
    # get seperate data of train and test
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    # scale the training and testing data
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)

    y_prob = predict_prob(X_train_scale, theta)

    print("Logreg train accuracy: %f" %
          (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    print("Logreg train precision: %f" %
          (sklearn.metrics.precision_score(y_train, y_prob > .5)))
    print("Logreg train recall: %f" %
          (sklearn.metrics.recall_score(y_train, y_prob > .5)))

    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" %
          (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    print("Logreg test precision: %f" %
          (sklearn.metrics.precision_score(y_test, y_prob > .5)))
    print("Logreg test recall: %f" %
          (sklearn.metrics.recall_score(y_test, y_prob > .5)))

    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)
