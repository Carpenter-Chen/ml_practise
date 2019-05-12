# coding=utf-8
import numpy as np

prob = 0.4
u_m = 172.5
s_m = 5

u_w = 155
s_w = 3


labels1 = np.random.normal(u_m, s_m, 6000)
labels2 = np.random.normal(u_w, s_w, 4000)


data = np.append(labels1, labels2)


u1 = 100
s1 = 20
u2 = 190
s2 = 30


def normal_prob(x, u, s):
    tmp1 = x - u
    tmp2 = - tmp1 * tmp1
    tmp3 = tmp2 / (2 * s)
    prob = np.exp(tmp3) / ((2 * np.pi * s) ** 0.5)
    return np.clip(prob, 0.0001, 1)


def E_step(x, parameter_list):
    result = []
    for p, u, s in parameter_list:
        prob_list = p * normal_prob(x, u, s)
        result.append(prob_list)

    dis = np.argmax(np.array(result), axis=0)
    new_prob = np.zeros((len(parameter_list), len(x)))
    new_prob[dis, range(len(x))] = 1
    return new_prob


def M_step(x, prob_list):
    result = []
    for each in prob_list:
        u = np.sum(x * each) / np.sum(each)
        s = np.sum((x - u) * (x - u) * each) / np.sum(each)
        p = np.average(each)
        result.append([p, u, s])
    return result


# 假设这里每个高斯分布有三个参数，一个是样本属于这个高斯分布的概率，一个均值，一个是方差
parameter_list = [[0.5, u1, s1], [0.5, u2, s2]]
for i in range(100):
    print "iteration: %d. %r" % (i, parameter_list)
    probs = E_step(data, parameter_list)
    parameter_list = M_step(data, probs)