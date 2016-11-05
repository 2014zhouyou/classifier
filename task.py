import os
import pickle
from classifiers import DecisionTreeClassifier
from classifiers import AdaBoostClassifier
from adult_data import AdultData


def test_holdout(adult_data, factor):
    # data filter
    train_set, test_set = adult_data.divide_holdout()
    print("data filtering....")
    print(train_set)
    print(len(train_set))
    print(test_set)
    print(len(test_set))
    labels = adult_data.get_labels()
    label_types = adult_data.get_label_types()
    print(labels)
    print(label_types)

    # train process
    if os.path.exists(factor + '_tree_holdout.pickle'):
        with open(factor + '_tree_holdout.pickle', 'rb') as f:
            clf = pickle.load(f)
    else:
        clf = DecisionTreeClassifier()
        clf.set_evaluation_factor(factor)
        clf.train(train_set, labels, label_types)
        with open(factor + '_tree_holdout.pickle', 'wb') as f:
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


def test_10_cross(adult_data, factor):
    """
    use 10 fold cross validation method to evaluate decision tree
    :param adult_data: data set to evaluate
    :return: None
    """
    labels = adult_data.get_labels()
    label_types = adult_data.get_label_types()
    data_set = adult_data.get_data_set()
    data_set_len = len(data_set)
    range_len = int((data_set_len - data_set_len % 10) / 10)
    data_set = data_set[0:10*range_len]
    clfs = []
    for i in range(0,10):
        print("training sub tree " + str(i))
        if os.path.exists(factor + '_10_cross.pickle_part'+str(i)):
            with open(factor + '_10_cross.pickle_part'+str(i), 'rb') as f:
                clf = pickle.load(f)
                clfs.append(clf)
        else:
            clf = DecisionTreeClassifier()
            train_set = get_train_set(i, range_len, data_set)
            clf.set_evaluation_factor(factor)
            clf.train(train_set, labels, label_types)
            clfs.append(clf)
            with open(factor + '_10_cross.pickle_part'+ str(i), 'wb') as f:
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


def test_bootstrap(adult_data, factor, times):
    data_set = adult_data.get_data_set()
    labels = adult_data.get_labels()
    label_types = adult_data.get_label_types()
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

        if os.path.exists(factor + "_tree_bootstrap.pickle_a" + str(i)):
            with open(factor + "_tree_bootstrap.pickle_a" + str(i), 'rb') as f:
                clf1 = pickle.load(f)
        else:
            clf1 = DecisionTreeClassifier()
            clf1.set_evaluation_factor(factor)
            clf1.train(train_set_a, labels, label_types)
            with open(factor + "_tree_bootstrap.pickle_a" + str(i), 'wb') as f:
                pickle.dump(clf1, f)

        if os.path.exists(factor + "_tree_bootstrap.pickle_b" + str(i)):
            with open(factor + "_tree_bootstrap.pickle_b" + str(i), 'rb') as f:
                clf2 = pickle.load(f)
        else:
            clf2 = DecisionTreeClassifier()
            clf2.set_evaluation_factor(factor)
            clf2.train(train_set_b, labels, label_types)
            with open(factor + "_tree_bootstrap.pickle_b" + str(i), 'wb') as f:
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
        print(factor + "the " + str(i) + " :correct count a = " + str(correct_count1))
        print(factor + "the " + str(i) + " :correct count b = " + str(correct_count2))
        length_of_test_set = len(test_set)
        print(factor + "test set length = " + str(length_of_test_set))
        accuracy = 0.632 * correct_count1 / length_of_test_set + 0.368 * correct_count2 / length_of_test_set
        print(factor + "accuracy = " + str(accuracy))

        accuracy_list.append(accuracy)

    print(factor + "bootstrap total " + str(times) + " :final accuracy: " + str((sum(accuracy_list) / times)))


def get_train_set(index, range_len, data_set):
    result = []
    result.extend(data_set[0:index * range_len])
    result.extend(data_set[(index + 1)* range_len:])
    return result


def test_pre_pruning():
    pass


def test_after_pruning():
    pass


def test_adaboost(adult_data):
    train_set, test_set = adult_data.divide_holdout()
    labels = adult_data.get_labels()
    label_types = adult_data.get_label_types()
    print("data processing")
    print(train_set)
    print(len(train_set))
    print(test_set)
    print(len(test_set))
    print(labels)
    print(label_types)

    print("training...")
    if os.path.exists('adaboost.pickle'):
        with open('adaboost.pickle', 'rb') as f:
            clf = pickle.load(f)
    else:
        clf = AdaBoostClassifier()
        clf.set_iterations(10)
        clf.train(train_set, labels, label_types)
        with open('adaboost.pickle', 'wb') as f:
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
    adult_data = AdultData('adult.csv')
    test_holdout(adult_data, 'GAIN')
    # test_10_cross(adult_data, 'GAIN')
    # test_bootstrap(adult_data, 'GAIN', 10)
    #
    # test_holdout(adult_data, 'GINI')
    # test_10_cross(adult_data, 'GINI')
    # test_bootstrap(adult_data, 'GINI', 10)

    # test_holdout(adult_data, 'MISCLASSFICATION')
    # test_10_cross(adult_data, 'MISCLASSFICATION')
    # test_bootstrap(adult_data, 'MISCLASSFICATION', 10)

    # test_pre_pruning()
    # test_after_pruning()

    # test_with_missing_value()

    # test_adaboost(adult_data)
