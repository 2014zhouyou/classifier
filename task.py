import os
import random
import pickle
from classifiers import DecisionTreeClassifier
from classifiers import AdaBoostClassifier
from data import AdultData
from data import IRisData
from data import CarData


def test_holdout(data, factor):
    base_dir = type(data).__name__ + '_cache/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # data filter
    # train_set, test_set = data.divide_holdout()
    data_set = data.get_data_set()
    # random.shuffle(data_set)
    len_data_set = len(data_set)
    train_set = data_set[0:int(0.7 *len_data_set)]
    test_set = data_set[int(0.7 * len_data_set):len_data_set]
    # test_set = data_set[0:len_data_set]
    labels = data.get_labels()
    label_types = data.get_label_types()
    print(train_set)
    print(test_set)
    print(labels)
    print(label_types)

    # train process
    if os.path.exists(base_dir + factor + '_tree_holdout.pickle'):
        with open(base_dir + factor + '_tree_holdout.pickle', 'rb') as f:
            clf = pickle.load(f)
    else:
        clf = DecisionTreeClassifier()
        clf.set_evaluation_factor(factor)
        clf.train(train_set, labels, label_types)

        with open(base_dir + factor + '_tree_holdout.pickle', 'wb') as f:
            pickle.dump(clf, f)
    # validation on the test set
    print(labels)
    correct_count = 0
    for row in test_set:
        predict_class = clf.fit_one(row, labels, label_types)
        if predict_class == row[-1]:
            correct_count += 1
    print("correct count = " + str(correct_count))
    length_of_test_set = len(test_set)
    print("test set length = " + str(length_of_test_set))
    accuracy = correct_count / length_of_test_set
    print("accuracy = " + str(accuracy))


def test_10_cross(data, factor):
    """
    use 10 fold cross validation method to evaluate decision tree
    :param data: data set to evaluate
    :return: None
    """
    base_dir = type(data).__name__ + '_cache/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    labels = data.get_labels()
    label_types = data.get_label_types()
    data_set = data.get_data_set()
    data_set_len = len(data_set)
    range_len = int((data_set_len - data_set_len % 10) / 10)
    data_set = data_set[0:10*range_len]
    clfs = list()
    for i in range(0,10):
        # print("training sub tree " + str(i))
        if os.path.exists(base_dir + factor + '_10_cross.pickle_part'+str(i)):
            with open(base_dir + factor + '_10_cross.pickle_part'+str(i), 'rb') as f:
                clf = pickle.load(f)
                clfs.append(clf)
        else:
            clf = DecisionTreeClassifier()
            train_set = get_train_set(i, range_len, data_set)
            clf.set_evaluation_factor(factor)
            clf.train(train_set, labels, label_types)
            clfs.append(clf)
            with open(base_dir + factor + '_10_cross.pickle_part'+ str(i), 'wb') as f:
                pickle.dump(clf, f)
    accuracy_list = []
    for i in range(0, 10):
        test_set = data_set[i * range_len:(i+1) * range_len]
        correct_count = 0
        for row in test_set:
            predict_class = clfs[i].fit_one(row, labels, label_types)
            if predict_class == row[-1]:
                correct_count += 1
        print("for classifier " + str(i) + ": ")
        print("correct count = " + str(correct_count))
        print("test set length = " + str(range_len))
        accuracy = correct_count / range_len
        print("accuracy = " + str(accuracy))
        accuracy_list.append(accuracy)
    print("average accuracy:" + str((sum(accuracy_list) / len(accuracy_list))))


def test_bootstrap(data, factor, times):
    base_dir = type(data).__name__ + '_cache/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    data_set = data.get_data_set()
    labels = data.get_labels()
    label_types = data.get_label_types()
    ada_clf = AdaBoostClassifier()
    num = len(data_set)
    accuracy_list = list()
    for i in range(0, times):
        print("times " + str(i))
        weights = [1 / num] * num
        sample_list = ada_clf.sample_with_weights(weights)
        train_set_a = [data_set[example] for example in sample_list]
        train_set_b = [data_set[example] for example in set(sample_list)]
        sample_remain_list = list(set([i for i in range(0, num)]).difference(sample_list))
        test_set = [data_set[example] for example in sample_remain_list]

        if os.path.exists(base_dir + factor + "_tree_bootstrap.pickle_a" + str(i)):
            with open(base_dir + factor + "_tree_bootstrap.pickle_a" + str(i), 'rb') as f:
                clf1 = pickle.load(f)
        else:
            clf1 = DecisionTreeClassifier()
            clf1.set_evaluation_factor(factor)
            clf1.train(train_set_a, labels, label_types)
            with open(base_dir + factor + "_tree_bootstrap.pickle_a" + str(i), 'wb') as f:
                pickle.dump(clf1, f)

        if os.path.exists(base_dir + factor + "_tree_bootstrap.pickle_b" + str(i)):
            with open(base_dir + factor + "_tree_bootstrap.pickle_b" + str(i), 'rb') as f:
                clf2 = pickle.load(f)
        else:
            clf2 = DecisionTreeClassifier()
            clf2.set_evaluation_factor(factor)
            clf2.train(train_set_b, labels, label_types)
            with open(base_dir + factor + "_tree_bootstrap.pickle_b" + str(i), 'wb') as f:
                pickle.dump(clf2, f)

        correct_count1 = 0
        for row in test_set:
            predict_class = clf1.fit_one(row, labels, label_types)
            if predict_class == row[-1]:
                correct_count1 += 1
        correct_count2 = 0
        for row in test_set:
            predict_class = clf2.fit_one(row, labels, label_types)
            if predict_class == row[-1]:
                correct_count2 += 1
        print(factor + " the " + str(i) + " :correct count a = " + str(correct_count1))
        print(factor + " the " + str(i) + " :correct count b = " + str(correct_count2))
        length_of_test_set = len(test_set)
        print(factor + " test set length = " + str(length_of_test_set))
        accuracy = 0.632 * correct_count1 / length_of_test_set + 0.368 * correct_count2 / length_of_test_set
        print(factor + " accuracy = " + str(accuracy))

        accuracy_list.append(accuracy)

    print(factor + " bootstrap total " + str(times) + " :final accuracy: " + str((sum(accuracy_list) / times)))


def get_train_set(index, range_len, data_set):
    result = []
    result.extend(data_set[0:index * range_len])
    result.extend(data_set[(index + 1)* range_len:])
    return result


def test_pre_pruning():
    pass


def test_after_pruning():
    pass


def test_adaboost(data):
    base_dir = type(data).__name__ + '_cache/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    data_set = data.get_data_set()
    # random.shuffle(data_set)
    len_data_set = len(data_set)
    train_set = data_set[0:int(0.7 * len_data_set)]
    test_set = data_set[int(0.7 * len_data_set):len_data_set]
    labels = data.get_labels()
    label_types = data.get_label_types()
    print("data processing")
    print("training...")
    if os.path.exists(base_dir + 'adaboost.pickle'):
        with open(base_dir + 'adaboost.pickle', 'rb') as f:
            clf = pickle.load(f)
    else:
        clf = AdaBoostClassifier()
        clf.set_iterations(10)
        clf.train(train_set, labels, label_types)
        with open(base_dir + 'adaboost.pickle', 'wb') as f:
            pickle.dump(clf, f)

    correct_count = 0
    all_class = set([example[-1] for example in test_set])
    for row in test_set:
        predict_class = clf.fit_one(row, all_class, labels, label_types)
        if predict_class == row[-1]:
            correct_count += 1
    print("correct count = " + str(correct_count))
    length_of_test_set = len(test_set)
    print("test set length = " + str(length_of_test_set))
    accuracy = correct_count / length_of_test_set
    print("accuracy = " + str(accuracy))

    for i in range (0, len(clf._clfs)):
        correct_count = 0
        for row in test_set:
            predict_class = clf._clfs[i][0].fit_one(row, labels, label_types)
            if predict_class == row[-1]:
                correct_count += 1
        print("correct count = " + str(correct_count))
        length_of_test_set = len(test_set)
        print("test set length = " + str(length_of_test_set))
        accuracy = correct_count / length_of_test_set
        print("accuracy = " + str(accuracy))


def test_with_missing_value():
    pass


if __name__ == '__main__':
    if os.path.exists('adult/adult.pickle'):
        with open('adult/adult.pickle', 'rb') as f:
            adult_data = pickle.load(f)
    else:
        adult_data = AdultData('adult/adult.csv')
        with open('adult/adult.pickle', 'wb') as f:
            pickle.dump(adult_data, f)
    # print(adult_data)

    if os.path.exists('iris/iris.pickle'):
        with open('iris/iris.pickle', 'rb') as f:
            iris_data = pickle.load(f)
    else:
        iris_data = IRisData('iris/iris.csv')
        with open('iris/iris.pickle', 'wb') as f:
            pickle.dump(iris_data, f)

    if os.path.exists('car/car.pickle'):
        with open('car/car.pickle', 'rb') as f:
            car_data = pickle.load(f)
    else:
        car_data = CarData('car/car.csv')
        with open('car/car.pickle', 'wb') as f:
            pickle.dump(car_data, f)

    test_holdout(adult_data, 'GAIN')
    test_10_cross(adult_data, 'GAIN')
    test_bootstrap(adult_data, 'GAIN', 10)
    test_holdout(iris_data, 'GAIN')
    test_10_cross(iris_data, 'GAIN')
    test_bootstrap(iris_data, 'GAIN', 10)
    test_holdout(car_data, 'GAIN')
    test_10_cross(car_data, 'GAIN')
    test_bootstrap(car_data, 'GAIN', 10)

    test_holdout(adult_data, 'GINI')
    test_10_cross(adult_data, 'GINI')
    test_bootstrap(adult_data, 'GINI', 10)
    test_holdout(iris_data, 'GINI')
    test_10_cross(iris_data, 'GINI')
    test_bootstrap(iris_data, 'GINI', 10)
    test_holdout(car_data, 'GINI')
    test_10_cross(car_data, 'GINI')
    test_bootstrap(car_data, 'GINI', 10)

    test_holdout(adult_data, 'MISCLASSFICATION')
    test_10_cross(adult_data, 'MISCLASSFICATION')
    test_bootstrap(adult_data, 'MISCLASSFICATION', 10)
    test_holdout(iris_data, 'MISCLASSFICATION')
    test_10_cross(iris_data, 'MISCLASSFICATION')
    test_bootstrap(iris_data, 'MISCLASSFICATION', 10)
    test_holdout(car_data, 'MISCLASSFICATION')
    test_10_cross(car_data, 'MISCLASSFICATION')
    test_bootstrap(car_data, 'MISCLASSFICATION', 10)

    # test_pre_pruning()
    # test_after_pruning()

    test_adaboost(adult_data)
