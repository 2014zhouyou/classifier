import csv
import copy
import random


class AdultData:
    data_set = []
    Name = "AdultDataSet"

    def __init__(self, data_file):
        source_file = open(data_file, 'r')
        reader = csv.reader(source_file)
        for row in reader:
            for i in range(0, len(row)):
                row[i] = row[i].strip()
            self.data_set.append(row)
        self._labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country', 'salary']
        NOMINAL = 'nominal'
        CONTINUOUS = 'continuous'
        self._label_types = [CONTINUOUS, NOMINAL, CONTINUOUS, NOMINAL, CONTINUOUS, NOMINAL,
                            NOMINAL, NOMINAL, NOMINAL, NOMINAL, CONTINUOUS, CONTINUOUS,
                            CONTINUOUS, NOMINAL, NOMINAL]

    def divide_holdout(self):
        copy_of_data = copy.deepcopy(self.data_set)
        # random.shuffle(copy_of_data)
        num_of_records = len(self.data_set)
        percentage = 0.65
        train_set = []
        for index in range(0, int(num_of_records * percentage)):
            train_set.append(copy_of_data[index])
        test_set = []
        for index in range(int(num_of_records * percentage), num_of_records):
            test_set.append(copy_of_data[index])
        return [train_set, test_set]

    def get_labels(self):
        return self._labels[:]

    def get_label_types(self):
        return self._label_types[:]

    def get_data_set(self):
        return self.data_set[:]

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

if __name__ == '__main__':
    test()
