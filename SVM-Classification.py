from sklearn import svm
import pandas as pd
import random

testdata = pd.read_csv('test_dataset.csv', header=None)
a2 = []
for i in testdata.index:
    a2.append(list(testdata.values[i]))
data_test = a2

traindata = pd.read_csv('dataset.csv', header=None)
a1 = []
for i in traindata.index:
    a1.append(list(traindata.values[i]))
data_train_set = a1
trainlabel = pd.read_csv('labelset.csv')
label_train_set= trainlabel['label']

class SVM_method:
    def __init__(self, train_num):
        self.train_num = train_num

    def choose_train_test_data(self):
        length_total = len(data_train_set) - 1
        test_data_num = []
        train_data_num = []
        test_data = []
        train_data = []
        test_label = []
        train_label = []
        moving_mark = 0
        while (moving_mark != self.train_num):
            num_random = random.randint(0, length_total)
            if train_data_num.count(num_random) == 0:
                train_data_num.append(num_random)
                moving_mark = moving_mark + 1
        for i in range(0, length_total):
            if train_data_num.count(i) == 0:
                test_data_num.append(i)
            else:
                continue
        for j in range(0, length_total):
            if test_data_num.count(j) != 0:
                test_data.append(data_train_set[j])
                test_label.append(label_train_set[j])
            elif train_data_num.count(j) != 0:
                train_data.append(data_train_set[j])
                train_label.append(label_train_set[j])
            else:
                print("Error happened. Please check the data.")
        return train_data, test_data, train_label, test_label, train_data_num, test_data_num

    def clf_function(self, gamma, C):
        clf = svm.SVC(kernel = 'rbf', gamma = gamma, C = C)
        return clf



    def Predict(self, predict_data, predict_true_label, gamma, C, train_data, train_label):
        accuracy_num = 0
        clf = self.clf_function(gamma, C)
        clf.fit(train_data, train_label)
        predict_label = clf.predict(predict_data)
        set_len = len(predict_data)
        for i in range(0, set_len):
            if predict_label[i] == predict_true_label[i]:
                accuracy_num = accuracy_num + 1
            else:
                continue
        accuracy_rate = (accuracy_num / set_len)
        return accuracy_rate, predict_label, clf

    def choose_gamma_C(self):
        gamma_list = []
        C_list = []
        step_length = 0.05
        for u in range(1, 100):
            current_number = u * step_length
            gamma_list.append(current_number)
            C_list.append(current_number)
        best_accuracy_rate = 0
        best_gamma = 0
        best_C = 0
        train_data, test_data, train_label, test_label, train_data_num, test_data_num = self.choose_train_test_data()
        for h in gamma_list:
            for g in C_list:
                accuracy_rate, predict_label, clf = self.Predict(test_data, test_label, h, g, train_data, train_label)
                if accuracy_rate >= best_accuracy_rate:
                    best_accuracy_rate = accuracy_rate
                    best_gamma = h
                    best_C = g
                    best_clf = clf
                else:
                    continue
        return best_accuracy_rate, best_gamma, best_C, best_clf

    def Just_to_print(self):
        accuracy_rate, gamma, C, clf = self.choose_gamma_C()
        print('The highest accuracy_rate is ', accuracy_rate)
        print('The most suitable gamma is ', gamma)
        print('The most suitable C is ', C)

    def get_prediction_label(self):
        accuracy_rate, gamma, C, clf = self.choose_gamma_C()
        predict_label = clf.predict(data_test)
        print(predict_label)


svm_result = SVM_method(231)
svm_result.Just_to_print()
print('The predictable labels of test data are')
svm_result.get_prediction_label()
