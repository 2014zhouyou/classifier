import csv
import copy
import random


class AdultData:
    # data_set = []
    # Name = "AdultDataSet"

    def __init__(self, data_file):
        source_file = open(data_file, 'r')
        reader = csv.reader(source_file)
        self.data_set = list()
        for row in reader:
            for i in range(0, len(row)):
                row[i] = row[i].strip()
            self.data_set.append(row)
        random.shuffle(self.data_set)
        self._labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country', 'salary']
        NOMINAL = 'nominal'
        CONTINUOUS = 'continuous'
        self._label_types = [CONTINUOUS, NOMINAL, CONTINUOUS, NOMINAL, CONTINUOUS, NOMINAL,
                             NOMINAL, NOMINAL, NOMINAL, NOMINAL, CONTINUOUS, CONTINUOUS,
                             CONTINUOUS, NOMINAL, NOMINAL]

    def get_labels(self):
        return self._labels[:]

    def get_label_types(self):
        return self._label_types[:]

    def get_data_set(self):
        return self.data_set[:]


class IRisData:
    def __init__(self, file_name):
        source_file = open(file_name, 'r')
        reader = csv.reader(source_file)
        self.data_set = list()
        for row in reader:
            self.data_set.append(row)
        random.shuffle(self.data_set)
        self._labels = ['sepal-length', 'sepal-with', 'petal-height', 'petal-with', 'class']
        NOMINAL = 'nominal'
        CONTINUOUS = 'continuous'
        self._label_types = [CONTINUOUS, CONTINUOUS, CONTINUOUS, CONTINUOUS, NOMINAL]

    def get_data_set(self):
        return self.data_set[:]

    def get_labels(self):
        return self._labels[:]

    def get_label_types(self):
        return self._label_types[:]


class CarData:
    def __init__(self, file_name):
        source_file = open(file_name, 'r')
        reader = csv.reader(source_file)
        self.data_set = list()
        for row in reader:
            self.data_set.append(row)
        random.shuffle(self.data_set)
        self._labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        NOMINAL = 'nominal'
        CONTINUOUS = 'continuous'
        self._label_types = [NOMINAL, NOMINAL, NOMINAL, NOMINAL, NOMINAL, NOMINAL]

    def get_data_set(self):
        return self.data_set[:]

    def get_labels(self):
        return self._labels[:]

    def get_label_types(self):
        return self._label_types[:]


def test():
    adult_data = AdultData('adult.csv')

    # test case 1
    # print(abalone_data.data_set)

    # test case 2
    data = adult_data.divide_holdout()
    print(len(data))
    print(len(data[0]))
    print(data[0])
    print(len(data[1]))
    print(data[1])

    # test case 3
    labels = adult_data.get_labels()
    print(labels)
    print(adult_data.get_label_types())


def test_iris_data():
    iris_data = IRisData('iris/iris.csv')
    print(iris_data.get_data_set())
    print(iris_data.get_labels())
    print(iris_data.get_label_types())


if __name__ == '__main__':
    # test()
    test_iris_data()