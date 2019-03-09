# coding: utf-8
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def soft_max(x):
    tot = sum(np.exp(x))
    return np.exp(x) / tot


class LSTM_Param():
    def __init__(self, dim_xc, dim_cell, dim_h, wrange=2):
        self.dim_xc = dim_xc
        self.dim_cell = dim_cell
        self.dim_h = dim_h

        # 三个门
        self.wf = np.random.rand(dim_cell, dim_xc) * wrange - wrange / 2
        self.wi = np.random.rand(dim_cell, dim_xc) * wrange - wrange / 2
        self.wo = np.random.rand(dim_h, dim_xc) * wrange - wrange / 2

        # 输入和输出的tanh
        self.wg = np.random.rand(dim_cell, dim_xc) * wrange - wrange / 2
        self.wc = np.random.rand(dim_h, dim_cell) * wrange - wrange / 2

        self.bf = np.random.rand(dim_cell) * wrange - wrange / 2
        self.bi = np.random.rand(dim_cell) * wrange - wrange / 2
        self.bo = np.random.rand(dim_h) * wrange - wrange / 2
        self.bg = np.random.rand(dim_cell) * wrange - wrange / 2
        self.bc = np.random.rand(dim_h) * wrange - wrange / 2

        self.reset_diff_parm()

    def update_parm(self):
        pass

    def reset_diff_parm(self):
        self.diff_wf = np.zeros((self.dim_cell, self.dim_xc))
        self.diff_wi = np.zeros((self.dim_cell, self.dim_xc))
        self.diff_wo = np.zeros((self.dim_h, self.dim_xc))
        self.diff_wg = np.zeros((self.dim_cell, self.dim_xc))
        self.diff_wc = np.zeros((self.dim_h, self.dim_cell))

        self.diff_bf = np.zeros(self.dim_cell)
        self.diff_bi = np.zeros(self.dim_cell)
        self.diff_bo = np.zeros(self.dim_h)
        self.diff_bg = np.zeros(self.dim_cell)
        self.diff_bc = np.zeros(self.dim_h)

    def apply_diff(self, learn_rate=0.03):

        clip_val = 50
        np.clip(self.diff_wf, -clip_val, clip_val)
        np.clip(self.diff_wi, -clip_val, clip_val)
        np.clip(self.diff_wg, -clip_val, clip_val)
        np.clip(self.diff_wo, -clip_val, clip_val)
        np.clip(self.diff_wc, -clip_val, clip_val)

        np.clip(self.diff_bf, -clip_val, clip_val)
        np.clip(self.diff_bi, -clip_val, clip_val)
        np.clip(self.diff_bg, -clip_val, clip_val)
        np.clip(self.diff_bc, -clip_val, clip_val)
        np.clip(self.diff_bo, -clip_val, clip_val)

        self.wf -= self.diff_wf * learn_rate
        self.wi -= self.diff_wi * learn_rate
        self.wg -= self.diff_wg * learn_rate
        self.wo -= self.diff_wo * learn_rate
        self.wc -= self.diff_wc * learn_rate

        self.bf -= self.diff_bf * learn_rate
        self.bi -= self.diff_bi * learn_rate
        self.bg -= self.diff_bg * learn_rate
        self.bo -= self.diff_bo * learn_rate
        self.bc -= self.diff_bc * learn_rate


class LSTM_Node():

    def __init__(self, x, y, lstm_parm, dim_cell, dim_x, dim_y):
        self.x = x
        self.y = y
        self.lstm_parm = lstm_parm
        self.dim_x = dim_x
        self.dim_xc = dim_x + dim_y
        self.dim_cell = dim_cell
        self.dim_h = dim_y

    def foward_propagation(self, prev_h, prev_CT):

        self.xc = np.hstack((prev_h, self.x))
        self.prev_CT = prev_CT

        self.zf = np.dot(self.lstm_parm.wf, self.xc) + self.lstm_parm.bf
        self.zi = np.dot(self.lstm_parm.wi, self.xc) + self.lstm_parm.bi
        self.zo = np.dot(self.lstm_parm.wo, self.xc) + self.lstm_parm.bo
        self.zg = np.dot(self.lstm_parm.wg, self.xc) + self.lstm_parm.bg

        self.af = sigmoid(self.zf)
        self.ai = sigmoid(self.zi)
        self.ag = tanh(self.zg)
        self.ao = sigmoid(self.zo)

        self.zct_ = self.ai * self.ag
        self.ct = self.af * self.prev_CT + self.zct_  # 上个cell state * 遗忘门 + 当前 输入门 * tanh(输入)
        self.zct = np.dot(self.lstm_parm.wc, self.ct) + self.lstm_parm.bc
        self.act = tanh(self.zct)

        self.h = self.ao * self.act
        self.sf_a = soft_max(self.h)

    def calc_logloss(self):
        return -np.log(max(self.sf_a))

    def delta_softmax_ce(self, label_y):
        # true_idx = np.argmax(label_y)
        # delta = self.sf_a * (-self.sf_a[true_idx])
        # delta[true_idx] = self.sf_a[true_idx] * (1 - self.sf_a[true_idx])
        delta = self.sf_a - label_y
        return delta

    def backword_propagation(self, next_delta_ct, delta_netxt_h):
        delta_y = self.delta_softmax_ce(self.y) + delta_netxt_h

        delta_ao = delta_y * self.act
        delta_act = delta_y * self.ao

        delta_zo = (1 - self.ao) * self.ao * delta_ao
        delta_zct = (1 - self.act ** 2) * delta_act

        self.lstm_parm.diff_wo += np.outer(delta_zo, self.xc)
        self.lstm_parm.diff_wc += np.outer(delta_zct, self.ct)
        self.lstm_parm.diff_bo += delta_zo
        self.lstm_parm.diff_bc += delta_zct

        delta_ct = next_delta_ct + np.dot(self.lstm_parm.wc.T, delta_zct)
        self.delta_prev_ct = delta_ct * self.af
        delta_af = delta_ct * self.prev_CT
        delta_ag = delta_ct * self.ai
        delta_ai = delta_ct * self.ag
        delta_zg = (1 - self.ag ** 2) * delta_ag
        delta_zi = (1 - self.ai) * self.ai * delta_ai
        delta_zf = (1 - self.af) * self.af * delta_af

        self.lstm_parm.diff_wi += np.outer(delta_zi, self.xc)
        self.lstm_parm.diff_wg += np.outer(delta_zg, self.xc)
        self.lstm_parm.diff_wf += np.outer(delta_zf, self.xc)

        self.lstm_parm.diff_bf += delta_zf
        self.lstm_parm.diff_bi += delta_zi
        self.lstm_parm.diff_bg += delta_zg

        delta_xc = np.dot(self.lstm_parm.wf.T, delta_zf)
        delta_xc += np.dot(self.lstm_parm.wi.T, delta_zi)
        delta_xc += np.dot(self.lstm_parm.wg.T, delta_zg)

        delta_xc += np.dot(self.lstm_parm.wi.T, delta_zi)

        self.delta_prev_h = delta_xc[:self.dim_h]


class LSTM_network():

    def __init__(self, dim_input, dim_y, dim_cell):
        self.dim_input = dim_input
        self.dim_y = dim_y
        self.dim_cell = dim_cell
        self.lstm_parm = LSTM_Param(dim_input + dim_y, dim_cell, dim_y)
        self.node_list = []

    def fit_one_record(self, x_list, y_list, apply_diff=True):
        self.each_node_process(x_list, y_list)
        # backword propagation
        for idx in range(len(self.node_list)):
            cur_idx = len(self.node_list) - idx - 1
            node = self.node_list[cur_idx]
            if idx == 0:
                delta_next_ct = np.zeros(self.dim_cell)
                delta_netxt_h = np.zeros(self.dim_y)
            else:
                delta_next_ct = self.node_list[cur_idx + 1].delta_prev_ct
                delta_netxt_h = self.node_list[cur_idx + 1].delta_prev_h
            node.backword_propagation(delta_next_ct, delta_netxt_h)

        if apply_diff:
            self.lstm_parm.apply_diff()
            self.lstm_parm.reset_diff_parm()

    def apply_diff(self):
        self.lstm_parm.apply_diff()
        self.lstm_parm.reset_diff_parm()

    def each_node_process(self, x_list, y_list):
        assert len(x_list) == len(y_list), "input num must equal to output"
        self.node_list = []
        for idx, (x, y) in enumerate(zip(x_list, y_list)):
            if idx == 0:
                prev_h = np.zeros(self.dim_y)
                prev_CT = np.zeros(self.dim_cell)
            else:
                prev_h = self.node_list[idx - 1].h
                prev_CT = self.node_list[idx - 1].ct
            node = LSTM_Node(x, y, self.lstm_parm, self.dim_cell, self.dim_input, self.dim_y)
            node.foward_propagation(prev_h, prev_CT)
            self.node_list.append(node)

    def calc_one_loss(self, x_list, y_list):
        self.each_node_process(x_list, y_list)
        tot_loss = 0
        for each in self.node_list:
            tot_loss += each.calc_logloss()

        return tot_loss / len(x_list)


def generate_data(dim_x=5, dim_y=10, len=6):
    cur_len = np.random.randint(len - 3, len + 3)
    x_list = [(np.random.rand(dim_x) * 2 - 1) for x in range(cur_len)]
    y_list = [np.zeros(dim_y) for x in range(cur_len)]
    for each in y_list:
        each[np.random.randint(0, dim_y)] = 1
    return x_list, y_list


def main():
    dim_x = 5  # 输入的维度
    dim_y = 10  # 输出的维度
    dim_cell = 10  # 细胞状态的维度
    np.random.seed(100)
    data = [generate_data(dim_x, dim_y) for x in range(10)]
    network = LSTM_network(dim_x, dim_y, dim_cell)
    for i in range(0, 10001):
        if i % 100 == 0:
            tot_loss = 0
            for x, y in data:
                tot_loss += network.calc_one_loss(x, y)
            print "iter %d err: %.5f" % (i, tot_loss / len(data))
        # for i in range(len(data)):
        #     import random
        #     rd = random.randint(0, len(data) - 1)
        #     x, y = data[rd]
        #     network.fit_one_record(x, y)
        for x, y in data:
            network.fit_one_record(x, y)
        # network.apply_diff()


if __name__ == "__main__":
    main()
