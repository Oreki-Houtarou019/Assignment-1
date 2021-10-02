import numpy as np
import math
import pandas as pd
traindata = pd.read_csv('dataset.csv', header=None)
a1 = []
for i in traindata.index:
    a1.append(list(traindata.values[i]))
data_train = a1
trainlabel = pd.read_csv('labelset.csv')
label_train = trainlabel['label']




def GaussianFunc(di, sigma):
    G = math.exp(-((di ** 2) / (2 * (sigma ** 2))))
    return G


def Neuron(centre, sigma, x):
    # distance=x-centre
    distance = list(map(lambda x: x[0] - x[1], zip(x, centre)))
    distance1 = []
    distance2 = 0
    dislen = len(distance)
    for i in range(0, dislen):
        m = distance[i] ** 2
        distance1.append(m)
    dis1len = len(distance1)
    for j in range(0, dis1len):
        distance2 = distance2 + distance1[j]
    distance2 = distance2 ** 0.5
    di = distance2
    oi = GaussianFunc(di, sigma)
    return oi


def train(centers, sigma):
    datalen = len(data_train)
    centrelen = len(centers)
    w_matrix = [[0 for j in range(0, centrelen + 1)] for i in range(0, datalen)]
    for i in range(0, datalen):
        for j in range(0, centrelen):
            w_matrix[i][j] = Neuron(centers[j], sigma, data_train[i])
    for k in range(0, datalen):
        w_matrix[k][centrelen] = 1
    w_matrix_trans = (np.mat(w_matrix)).T
    p_matrix = w_matrix
    w_matrix = np.dot(np.dot(np.linalg.inv(np.dot(w_matrix_trans, w_matrix)), w_matrix_trans), label_train)
    w_matrix = (np.mat(w_matrix)).T
    return p_matrix, w_matrix


def Predict(p_matrix, w_matrix):
    p_label = np.dot(p_matrix, w_matrix)
    return p_label


def Accuracy_Rate(real_label, p_label):
    real_label = label_train
    p_l_len = len(p_label)
    r_l_len = len(real_label)
    true_point = 0
    for j in range(0, p_l_len):
        if p_label[j] > 0:
            p_label[j] = 1
        else:
            p_label[j] = -1

    for i in range(0, p_l_len):
        if p_label[i] == real_label[i]:
            true_point = true_point + 1
    accuracy_rate = (true_point / r_l_len)
    return accuracy_rate


def Choose_center():
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    max_part = 0
    max_total = 0
    max_maximum = 0
    centers= []
    moving_list = []
    value_list = []
    value_list_total = []
    centers_proper = []
    num_centers_list = []
    for i in range(0, 9):
        sigma = sigma_list[i]
        centers.clear()
        value_list.clear()

        for j in range(0,2):
            for k in range(0,len(a1)-1):
                centers.append(a1[k])
                p_matrix, w_matrix = train(centers, sigma)
                p_label = Predict(p_matrix, w_matrix)
                accuracy_rate = Accuracy_Rate(label_train, p_label)
                centers.pop()
                if accuracy_rate > max_part:
                    max_part = accuracy_rate
                    moving_list.append(k)
                if k == (len(a1)-1):
                    abon = moving_list.pop()
                    centers.append(a1[abon])
                    del a1[abon]
                    value_list.append(max_part)
                    moving_list.clear()
            if max_part > max_total:
                max_total = max_part
                num_centers = j
                centers_proper.append(centers)
        value_list_total.append(value_list)
        num_centers_list.append(num_centers)
        if max_total > max_maximum:
            max_maximum = max_total
            sigma_max = sigma_list[i]

    return centers_proper, value_list_total, max_maximum, num_centers_list, sigma_max


centers_proper, value_list_total, accuracy_rate_max, num_centers_list, sigma_max = Choose_center()
print(centers_proper)
print(value_list_total)
print(accuracy_rate_max)
print(num_centers_list)
print(sigma_max)





