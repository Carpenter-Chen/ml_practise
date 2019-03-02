# coding: utf-8
from utils.data_loader import load_iris_data_for_lr
import numpy as np
import random as rd
import math
alpha = 0.02

def load_data():
    train, test = load_iris_data_for_lr()
    return train, test


def sigmoid_func(x):
    return min(0.9999, max(0.0001, 1 / (1 + math.exp(-x))))


def sgd(data_set, w):
    data_num, feature_num = data_set.shape
    random_data = data_set[rd.randint(0, data_num - 1)]
    pred = sigmoid_func(random_data[:-1].dot(w))
    label = random_data[-1]
    w = w + alpha * (label - pred) * random_data[:-1]
    return w


def eval(test_data, w, data_name):
    result = test_data[:, :-1].dot(w)
    label = test_data[:, -1]
    logloss = 0
    for p, y in zip(result, label):
        logloss += y * math.log(sigmoid_func(p)) + (1 - y) * math.log((1 - sigmoid_func(p)))
    print "##########data:%s logloss: %.4f" % (data_name, logloss / len(result))


def main():
    train, test = load_data()
    w_length = len(train[0]) - 1
    w = np.array([rd.random() for i in range(w_length)], dtype=float)
    loop_times = 2000
    for i in range(loop_times):
        w = sgd(train, w)
        if i % 100 == 1:
            eval(train, w, 'train')
            eval(test, w, 'test')


if __name__ == "__main__":
    main()
