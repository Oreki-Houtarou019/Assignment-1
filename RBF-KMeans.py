import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
traindata = pd.read_csv('dataset.csv', header=None)
a1 = []
for i in traindata.index:
    a1.append(list(traindata.values[i]))
data_train = a1
testdata = pd.read_csv('test_dataset.csv', header=None)
a2 = []
for i in testdata.index:
    a2.append(list(testdata.values[i]))
data_test = a2
trainlabel = pd.read_csv('labelset.csv')
label_train = trainlabel['label']


class RBF:
    def __init__(self, num_centers):
        self.num_centers = num_centers



    def center_selection(self):
        datalen = len(data_train)
        number = []
        centers = []
        distance1 = []
        max_part = 0
        max_total = 0
        i = 0
        kmeans = KMeans(n_clusters=self.num_centers, random_state=0).fit(data_train)
        centers = kmeans.cluster_centers_

        for i in range(0, self.num_centers):
            for j in range(i, self.num_centers):
                distance = list(map(lambda x: x[0] - x[1], zip(centers[i], centers[j])))
                distance1 = []
                dislen = len(distance)

                for m in range(0, dislen):
                    distance1.append(distance[m] ** 2)
                dis1len = len(distance1)
                for n in range(0, dis1len):
                    max_part = max_part + distance1[n]
                max_part = max_part ** 0.5
                if max_part > max_total:
                    max_total = max_part
        dmax = max_total
        sigma = (dmax / ((2 * self.num_centers) ** 0.5))
        return centers, sigma

    def GaussianFunc(self, di, sigma):
        G = math.exp(-((di ** 2) / (2 * (sigma ** 2))))
        return G

    def Neuron(self, centre, sigma, x):
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
        oi = self.GaussianFunc(di, sigma)
        return oi

    def train(self):
        centers, sigma = self.center_selection()
        datalen = len(data_train)
        centrelen = len(centers)
        w_matrix = [[0 for j in range(0, centrelen + 1)] for i in range(0, datalen)]
        for i in range(0, datalen):
            for j in range(0, centrelen):
                w_matrix[i][j] = self.Neuron(centers[j], sigma, data_train[i])
        for k in range(0, datalen):
            w_matrix[k][centrelen] = 1
        w_matrix_trans = (np.mat(w_matrix)).T
        p_matrix = w_matrix
        w_matrix = np.dot(np.dot(np.linalg.inv(np.dot(w_matrix_trans, w_matrix)), w_matrix_trans), label_train)
        w_matrix = (np.mat(w_matrix)).T
        return p_matrix, w_matrix

    def Predict(self):
        p_matrix, w_matrix = self.train()
        p_label = np.dot(p_matrix, w_matrix)
        return p_label

    def Accuracy_Rate(self):
        real_label = label_train
        p_label = self.Predict()
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

    def Just_to_print(self):
        centers, sigma = self.center_selection()
        print(sigma)

    def get_predict_test(self):
        centers, sigma = self.center_selection()
        datalen = len(data_test)
        centrelen = len(centers)
        w_matrix = [[0 for j in range(0, centrelen + 1)] for i in range(0, datalen)]
        for i in range(0, datalen):
            for j in range(0, centrelen):
                w_matrix[i][j] = self.Neuron(centers[j], sigma, data_test[i])
        for k in range(0, datalen):
            w_matrix[k][centrelen] = 1
        p_matrix_test = w_matrix
        a, w_matrix_test = self.train()
        p_label_test = np.dot(p_matrix_test, w_matrix_test)
        p_l_t_len = len(p_label_test)
        for j in range(0, p_l_t_len):
            if p_label_test[j] > 0:
                p_label_test[j] = 1
            else:
                p_label_test[j] = -1
        print(p_label_test)
        return p_label_test

value_max = 0
for i in range(5, 150, 5):
    result = RBF(i)
    value_tempo = result.Accuracy_Rate()
    if value_tempo > value_max:
        value_max = value_tempo
        num_proper = i
        result_best = result
    else:
        continue

print('The highest accuracy rate is ' , value_max * 100, '%')
print('The most suitable number of neurons is ', num_proper)
print('The sigma is ')
result_best.Just_to_print()
print('The predictable labels of test data are')
result_best.get_predict_test()
