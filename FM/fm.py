# code=utf8
import math
import numpy as np
import random as rd


def sigmoid(x):
        return 1.0 / (1 + math.exp(-x))


def logloss(px, y):
    return -math.log(px) if y > 0 else -math.log(1 - px)


class FM():

    def __init__(self, n_feature, k=10):
        self.w0 = rd.random()
        self.w = np.random.random(n_feature)
        self.latent_w = np.random.random([n_feature, k])
        self.alpha = 0.005
        self.learning_rate = 0.02


    def sgd(self, train_data):
        for i in range(len(train_data)):
            cur_sample = train_data[i, :-1]
            label = train_data[i, -1]
            result = self.inference_one_sample(cur_sample)
            y_hat = sigmoid(result)
            self.w0 = self.w0 - self.learning_rate * (y_hat - label)
            self.w = self.w - self.learning_rate * (y_hat - label) * cur_sample
            prefix_loss = self.alpha * (y_hat - label)
            for j in range(len(self.latent_w)):
                self.latent_w[j] -= self.learning_rate * prefix_loss * (np.dot(cur_sample, self.latent_w) * cur_sample[j] - self.latent_w[j] * cur_sample[j] * cur_sample[j])


    def inference_one_sample(self, data):
        assert len(data) == len(self.w), "invaild data length"
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        sum1_array = np.dot(data, self.latent_w)
        sum1 = sum1_array.dot(sum1_array)

        sum2_array = sum1_array * sum1_array
        sum2 = sum(sum2_array)

        result = self.w0 + data.dot(self.w) + 0.5 * (sum1 - sum2)
        return result

    def calc_mean_loss(self, validation_data):
        tot = 0
        for each in validation_data:
            data = each[:-1]
            y = each[-1]
            tot += logloss(sigmoid(self.inference_one_sample(data)), y)

        return tot / len(validation_data)


    def train(self, train_data, validation_data, iteration=10):
        for iter_num in range(iteration):
            mean_loss = self.calc_mean_loss(validation_data)
            print "logloss: ", mean_loss
            self.sgd(train_data)

if __name__ == "__main__":
    from utils.data_loader import load_iris_data_for_lr
    train, test = load_iris_data_for_lr()
    fm = FM(len(train[0]) - 1)
    fm.train(train, test)