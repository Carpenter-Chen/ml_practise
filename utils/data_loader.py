# coding: utf-8
import numpy as np
import os
import random as rd
IRIS_DATA_NAME = 'iris.data'


def get_cur_dir():
    u"""获取当前文件的路径."""
    return os.path.abspath(os.path.dirname(__file__))


def get_data_abs_path(file_path):
    return os.path.join(get_cur_dir(), "../data/%s" % file_path)


def load_iris_data_for_lr(target_class_list=[0, 1], test_data_ratio=0.2):
    u"""加载iris的数据用于lr分类器，iris数据有三个类目，target_class_list指定."""
    category_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    file_path = get_data_abs_path(IRIS_DATA_NAME)
    tot_data = []
    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            if len(items) < 5:
                print "Invaild Line for iris data"
                continue
            label = items[-1]
            if label not in category_mapping or category_mapping[label] not in target_class_list:
                continue
            label_id = category_mapping[label]
            tot_data.append(items[:-1] + [1, label_id])

    tot_data_set = np.array(tot_data, dtype=float)
    data_num = len(tot_data_set)
    test_index = filter(lambda x: rd.random() < test_data_ratio, range(data_num))
    test_index_set = set(test_index)
    train_index = filter(lambda x: x not in test_index_set, range(data_num))
    train_data_set = tot_data_set[train_index]
    test_data_set = tot_data_set[test_index]
    return train_data_set, test_data_set
