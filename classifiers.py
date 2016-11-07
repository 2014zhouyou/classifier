import math
import random
import time
from data import AdultData


class DecisionTreeClassifier:
    GAIN = 'GAIN'
    GINI = 'GINI'
    MISCLASSFICATION = 'MISCLASSFICATION'
    _CONTINUOUS = 'continuous'
    _NOMINAL = 'nominal'
    _VALUE_NOT_EXIST = 'NOT EXIST'
    _EVALUATION_METHOD_MAP = None
    _tree = None
    _factor = 'GAIN'

    def __init__(self):
        self._EVALUATION_METHOD_MAP = dict()
        self._EVALUATION_METHOD_MAP[self.GAIN] = self._gain_best_split
        self._EVALUATION_METHOD_MAP[self.GINI] = self._gini_best_split
        self._EVALUATION_METHOD_MAP[self.MISCLASSFICATION] = self._error_best_split

    def set_evaluation_factor(self, value):
        self._factor = value

    def train(self, data_set, labels, label_types):
        """
        train a decision tree
        :param data_set: [[x1, x2, x3, ..., y], ...]
        :param labels: class labels
        :param ev_fac: evaluation factor, three possible input: 'gain', 'gini', 'error', default is 'gain'
        :return:
        """
        # self.count = 0
        self._tree = self._compute_tree(data_set, labels, label_types)

    def fit_one(self, x, label, label_types):
        """
        to output the class of the input record using the trained tree
        :param record: the record you want to classify, type: Map
        :return: an integer number represent the class
        """
        return self._fit(self._tree, x, label, label_types)

    def fit(self, X, label, label_types):
        result = list()
        for item in X:
            result.append(self.fit_one(item, label, label_types))
        return result

    def _fit(self, tree, record, labels, label_types):
        """
        traversal down the tree, to classify the record
        :param tree: the decision tree
        :param record: the record you want to classify
        :return: the class of the record
        """
        if type(tree) == type(''):
            return tree
        elif type(tree) == type({}):
            if len(tree.keys()) == 0:
                print("Error, the key of the tree should not be null!")
            key = list(tree.keys())[0]
            index_of_label = labels.index(key)
            value = record[index_of_label]
            if label_types[index_of_label] == self._NOMINAL:
                # for nominal attribute, there is the case that it doesn't have the key in the subtree
                if value in tree[key].keys():
                    return self._fit(tree[key][value], record, labels, label_types)
                else:
                    return self._fit(tree[key][self._VALUE_NOT_EXIST], record, labels, label_types)
            # elif value[0].isdigit():
            elif label_types[index_of_label] == self._CONTINUOUS:
                division_point = self._get_division_point(list(tree[key])[0])
                # print(list(tree[key]))
                if float(value) <= float(division_point):
                    return self._fit(tree[key]['<=' + str(division_point)], record, labels, label_types)
                else:
                    return self._fit(tree[key]['>' + str(division_point)], record, labels, label_types)
            else:
                print("Error the attribute should be nominal or continuous")
                return None

    def _get_division_point(self, value):
        """
        get the the division point
        :param value: the label
        :return: a number represent the division point
        """
        # print("nanai" + str(value))
        if len(value) == 0:
            print("Error, the value should not be null")
        result = ''
        for i in range(0, len(value)):
            if value[i].isdigit() or value[i] == '.':
                result = result + value[i]
        # print("god" + str(len(result)))
        # print(result)
        return result

    def _compute_tree(self, ori_data_set, ori_labels, ori_label_types):
        """Generate a decision tree for the given data set
                :param data_set: the training data
                :param labels: the attribute name
                :return: the generated decision tree
                """
        # self.count += 1
        data_set = ori_data_set[:]
        labels = ori_labels[:]
        label_types = ori_label_types[:]
        class_list = [example[-1] for example in data_set]
        # when all the record in the data set has the same class, stop this
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        # when there is no more attribute can be use in the data set, stop this
        if len(data_set[0]) == 1 or self._check_same_attribute(data_set):
            return self._majority_vost(class_list)
        # step 1 choose one attribute to split
        choose_method = self._EVALUATION_METHOD_MAP[self._factor]
        best_value, best_feature, seperating_value = choose_method(data_set, label_types)
        # when a data set in a node has the same distribution, like this:
        # [['Private', '13', '<=50K'], ['Private', '13', '>50K'],
        #  ['Self-emp-inc', '13', '>50K'], ['Private', '13', '<=50K'],
        #  ['Self-emp-inc', '13', '<=50K'], ['Private', '13', '>50K']]
        # split the data set won't help increasing the accuracy and cause something difficut to handle
        # so it should be stopped This is the one case.
        # the index no help function handle many cases that should be stopped.
        if self._index_no_help(self._factor, best_value):
            return self._majority_vost(class_list)

        best_label = labels[best_feature]
        best_label_type = label_types[best_feature]
        mytree = {best_label: {}}
        del labels[best_feature]
        del label_types[best_feature]
        # step 2 divide data into parts according to the best split attribute and repeat process
        if best_label_type == self._NOMINAL:
            best_feature_values = [example[best_feature] for example in data_set]
            unique_best_feature_values = set(best_feature_values)
            for value in unique_best_feature_values:
                data_set_part = self._find_record_with_value(best_feature, data_set, value)
                sub_labels = labels[:]
                sub_label_types = label_types[:]
                data_set_part = self._remove_feature(best_feature, data_set_part)
                mytree[best_label][value] = self._compute_tree(data_set_part, sub_labels, sub_label_types)
            mytree[best_label][self._VALUE_NOT_EXIST] = self._majority_vost(class_list)
        elif best_label_type == self._CONTINUOUS:
            under_point_data, over_point_data = self._split_data(best_feature, data_set, seperating_value)
            """
            print("xxxx" + str(best_feature))
            print("xxxx" + str(under_point_data))
            print("xxxx" + str(len(under_point_data)))
            print("xxxx" + str(over_point_data))
            print("xxxx" + str(len(over_point_data)))
            print("xxxx" + str(seperating_value))
            """
            ori_under_point_data = under_point_data[:]
            ori_over_point_data = over_point_data[:]
            if 0 < len(under_point_data):
                under_point_data = self._remove_feature(best_feature, under_point_data)
                label_for_small_data = '<=' + str(seperating_value)
                sub_labels1 = labels[:]
                sub_label1_types = label_types[:]
                mytree[best_label][label_for_small_data] = self._compute_tree(under_point_data, sub_labels1,
                                                                              sub_label1_types)
            else:
                print("Attention, left sub tree is NULL!This is the error case may not happen!")
                print("xxxx" + str(best_feature))
                print("xxxx" + str(seperating_value))
                print("Right data is :" + str(ori_over_point_data))

            # del over_point_data[best_feature]
            if 0 < len(over_point_data):
                over_point_data = self._remove_feature(best_feature, over_point_data)
                label_for_larger_data = '>' + str(seperating_value)
                sub_labels2 = labels[:]
                sub_label2_types = label_types[:]
                mytree[best_label][label_for_larger_data] = self._compute_tree(over_point_data, sub_labels2,
                                                                               sub_label2_types)
            else:
                print("Attention, right sub tree is NULL, This is the error case!")
                print("yyyy" + str(best_feature))
                print("yyyy" + str(seperating_value))
                print("left data is :" + str(ori_under_point_data))
        else:
            print('ERROR!!!! Attention Please!!!!')
        return mytree

    def _index_no_help(self, factor, value):
        if factor == self.GAIN:
            if abs(value) <= 0:
                return True
            else:
                return False
        elif factor == self.GINI:
            return False
        else:
            return False

    def _check_same_attribute(self, data_set):
        """
        to see if the record has all the same attribute value in the data set
        :param data_set: the data set you want to check, not null
        :return:True, all the record in the data set have same attribute value, else False
        """
        if len(data_set) == 0:
            print("data set is null!")

        data_copy = [example[:-1] for example in data_set]
        attribute_num = len(data_copy[0])
        base_record = data_copy[0]
        for row in data_copy:
            for i in range(0, attribute_num):
                if base_record[i] != row[i]:
                    return False
        return True

    def _remove_feature(self, index, data_set):
        result_set = []
        record_len = len(data_set[0])
        for i in range(0, len(data_set)):
            row = data_set[i]
            temp = row[0:index]
            temp.extend(row[index+1:record_len])
            result_set.append(temp)
        return result_set

    def _majority_vost(self, class_list):
        """
        select the most common case of class
        :param class_list: a list of class
        :return:
        """
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        result = None
        num = 0
        for key in class_count.keys():
            if class_count[key] >= num:
                result = key
                num = class_count[key]
        return result

    def _find_record_with_value(self, feature, data_set, value):
        """
        find some data whose the feature value equals to the given value in the data set
        :param feature: the index of the record
        :param data_set: the whole data set
        :param value: the value you care about
        :return: a list of record whose feature attribute equals value
        """
        target_data_set = []
        for row in data_set:
            if row[feature] == value:
                target_data_set.append(row)
        return target_data_set

    def _split_data(self, feature, data_set, division_point):
        """
        split the data set into two part for continuous attribute according to the division point
        :param feature: index of the feature
        :param data_set: the whole data set you want to split
        :param division_point: the point
        :return: [x, y], x is the data set  whose feature value is smaller than the division point
                 y is the data set whose feature value is larger than the division point
        """
        under_point_data = []
        over_point_data = []
        for row in data_set:
            if float(row[feature]) <= division_point:
                under_point_data.append(row)
            elif division_point < float(row[feature]):
                over_point_data.append(row)
            else:
                print("Error!!!!!!Attention Please!!!")
        # if len(under_point_data) == 0:
        #    print("in split" + str(under_point_data))
        #    print("in split" + str(over_point_data))
        #    print("in split" + str(division_point))
        return [under_point_data, over_point_data]

    def _calc_entropy(self, data_set):
        """
        compute the entropy of the data_set
        :param data_set: the data_set you want to calculator entropy
        :return: x, x is a numerical value represent the entropy
        """
        class_list = [example[-1] for example in data_set]
        unique_class = set(class_list)
        record_num = len(data_set)
        entropy = 0.0
        for item in unique_class:
            item_data = self._find_record_with_value(-1, data_set, item)
            probability = len(item_data) / record_num
            entropy -= probability * math.log2(probability)
        return entropy

    def _gain_best_split(self, data_set, label_types):
        """find best split attribute
        :param data_set: the data set at node t
        :return: [g, x, y], x is the best attribute index, if the attribute is a continuous attribute
        y is the division point to split the data into two part, else y = -1, g is the corresponding best gain
        """
        # traversal all attribute
        num_of_attributes = len(data_set[0]) - 1
        num_of_record = len(data_set)
        best_gain = 0.0
        best_feature = -1
        best_divison_arg = -1
        base_entropy = self._calc_entropy(data_set)
        for i in range(0, num_of_attributes):
            gain = 0.0
            seperating_value = 0.0
            attr_values = [example[i] for example in data_set]
            # if the attr is nominal
            if label_types[i] == self._NOMINAL:
                # divide the data into n part, which n is  unique value of the attribute
                unique_values = set(attr_values)
                # compute the gain of the split as v
                for value in unique_values:
                    data_contains_value = self._find_record_with_value(i, data_set, value)
                    gain -= len(data_contains_value) / num_of_record * self._calc_entropy(data_contains_value)
                gain = gain + base_entropy
                seperating_value = 0.0
            # else if the attr is continuous
            elif label_types[i] == self._CONTINUOUS:
                # sort the unique value according the attribute
                unique_values = list(set(attr_values))
                values_to_num = [float(example) for example in unique_values]
                values_to_num.sort()
                local_best_gain = 0.0
                seperating_value = values_to_num[0] # handle the case that there is the only one
                # gen k division point, setting k according to the unique value count subtract one
                for j in range(0, len(values_to_num) - 1):
                    # for every division point k, compute the gain, and find the highest gain as v and
                    # the corresponding k
                    gain = 0.0
                    division_point = (values_to_num[j] + values_to_num[j+1]) / 2
                    under_point_data, over_point_data = self._split_data(i, data_set, division_point)
                    gain -= len(under_point_data) / num_of_record * self._calc_entropy(under_point_data)
                    gain -= len(over_point_data) / num_of_record * self._calc_entropy(over_point_data)
                    gain += base_entropy
                    if local_best_gain <= gain:
                        local_best_gain = gain
                        seperating_value = division_point
                gain = local_best_gain
            # compete the current attribute's v and last highest pv
            # if the v >= pv, update the highest v, and best attribute
            if best_gain <= gain:
                best_gain = gain
                best_feature = i
                best_divison_arg = seperating_value
        return [best_gain, best_feature, best_divison_arg]

    def _calc_gini(self, data_set):
        nums_record = len(data_set)
        gini = 1
        class_list = [example[-1] for example in data_set]
        unique_class = set(class_list)
        for value in unique_class:
            probability = class_list.count(value) / nums_record
            gini -= probability * probability
        return gini

    def _gini_best_split(self, data_set, label_types):
        # traversal all attribute
        nums_of_attr = len(data_set[0]) - 1
        nums_of_record = len(data_set)
        best_gini = 1.0
        best_divison_arg = -1
        best_feature = -1
        for i in range(0, nums_of_attr):
            gini = 0.0
            seperating_value = 0.0
            attr_values = [example[i] for example in data_set]
            # if the attr is nominal
            if label_types[i] == self._NOMINAL:
                # divide the data into n part, which n is  unique value of the attribute
                unique_values = set(attr_values)
                # compute the gini of the split as v
                for value in unique_values:
                    data_contains_value = self._find_record_with_value(i, data_set, value)
                    gini += len(data_contains_value) / nums_of_record * self._calc_gini(data_contains_value)
                seperating_value = 0.0
            # else if the attr is continuous
            elif label_types[i] == self._CONTINUOUS:
                # sort the unique value according the attribute
                unique_values = list(set(attr_values))
                values_to_num = [float(example) for example in unique_values]
                values_to_num.sort()
                local_best_gini = 1.0
                seperating_value = values_to_num[0]  # handle the case that there is the only one
                # gen k division point, setting k according to the unique value count subtract one
                for j in range(0, len(values_to_num) - 1):
                    # for every division point k, compute the gain, and find the highest gain as v and
                    # the corresponding k
                    gini = 0.0
                    division_point = (values_to_num[j] + values_to_num[j + 1]) / 2
                    under_point_data, over_point_data = self._split_data(i, data_set, division_point)
                    gini += len(under_point_data) / nums_of_record * self._calc_gini(under_point_data)
                    gini += len(over_point_data) / nums_of_record * self._calc_gini(over_point_data)
                    if gini <= local_best_gini:
                        local_best_gini = gini
                        seperating_value = division_point
                gini = local_best_gini
            # compete the current attribute's v and last highest pv
            # if the v >= pv, update the highest v, and best attribute
            if gini <= best_gini:
                best_gini = gini
                best_feature = i
                best_divison_arg = seperating_value
        return [best_gini, best_feature, best_divison_arg]

    def _calc_error(self, data_set):
        nums_record = len(data_set)
        class_list = [example[-1] for example in data_set]
        unique_class = set(class_list)
        probability_list = []
        for value in unique_class:
            probability = class_list.count(value) / nums_record
            probability_list.append(probability)
        error = 1 - max(probability_list)
        return error

    def _error_best_split(self, data_set, label_types):
        # no matter what kind of data set is left, if they have different attribute,
        # error can select the different one
        # traversal all attribute
        nums_of_attr = len(data_set[0]) - 1
        nums_of_record = len(data_set)
        best_error = 1.0
        best_divison_arg = -1
        best_feature = -1
        for i in range(0, nums_of_attr):
            error = 0.0
            seperating_value = 0.0
            attr_values = [example[i] for example in data_set]
            # if the attr is nominal
            if label_types[i] == self._NOMINAL:
                # divide the data into n part, which n is  unique value of the attribute
                unique_values = set(attr_values)
                # compute the gini of the split as v
                for value in unique_values:
                    data_contains_value = self._find_record_with_value(i, data_set, value)
                    error += len(data_contains_value) / nums_of_record * self._calc_error(data_contains_value)
                seperating_value = 0.0
            # else if the attr is continuous
            elif label_types[i] == self._CONTINUOUS:
                # sort the unique value according the attribute
                unique_values = list(set(attr_values))
                values_to_num = [float(example) for example in unique_values]
                values_to_num.sort()
                local_best_error = 1.0
                seperating_value = values_to_num[0]  # handle the case that there is the only one
                # gen k division point, setting k according to the unique value count subtract one
                for j in range(0, len(values_to_num) - 1):
                    # for every division point k, compute the gain, and find the highest gain as v and
                    # the corresponding k
                    error = 0.0
                    division_point = (values_to_num[j] + values_to_num[j + 1]) / 2
                    under_point_data, over_point_data = self._split_data(i, data_set, division_point)
                    error += len(under_point_data) / nums_of_record * self._calc_gini(under_point_data)
                    error += len(over_point_data) / nums_of_record * self._calc_gini(over_point_data)
                    if error <= local_best_error:
                        local_best_error = error
                        seperating_value = division_point
                error = local_best_error
            # compete the current attribute's v and last highest pv
            # if the v >= pv, update the highest v, and best attribute
            if error <= best_error:
                best_error = error
                best_feature = i
                best_divison_arg = seperating_value
        return [best_error, best_feature, best_divison_arg]

    def _pre_pruning(self, data_set):
        pass

    def _post_pruning(self, data_set):
        pass


class AdaBoostClassifier:
    DEFAULT_ITERATIONS = 10

    def __init__(self):
        self._clfs = list()
        self._iterations = self.DEFAULT_ITERATIONS

    def set_iterations(self, value):
        self._iterations = value

    def sample_with_weights(self, weights):
        record_num = len(weights)
        boundary_table = list()
        boundary_table.append(0)
        sum_boundary = 0.0
        for i in range(0, record_num):
            sum_boundary = sum_boundary + weights[i]
            boundary_table.append(sum_boundary)
        scale_ratio = 100000000
        num = scale_ratio * record_num
        scale_boundary_table = [int(num * example) for example in boundary_table]
        max_boundary = scale_boundary_table[-1]
        sample_record = list()
        for i in range(0, record_num):
            num = random.randint(0, max_boundary)
            index= self._binsearch(scale_boundary_table, num, 0, record_num)
            sample_record.append(index)
        return sample_record

    def _binsearch(self, target_list, num, left, right):
        if right <= left + 1 and (num < target_list[left] or target_list[right] < num):
            print("cant' found, attention please")
            return -1
        if right <= left + 1:
            return left
        middle = int((left + right) / 2)
        if num <= target_list[middle]:
            return self._binsearch(target_list, num, left, middle)
        else:
            return self._binsearch(target_list, num, middle, right)

    def train(self, data_set, label, label_types):
        self._train(data_set, label, label_types)

    def _train(self, data_set, label, label_types):
        self._clfs.clear()
        my_data_set = data_set[:]
        record_num = len(my_data_set)
        weights = [1 / record_num] * record_num
        for i in range(0, self._iterations):
            print("training times: " + str(i))
            sample_list = self.sample_with_weights(weights)
            print(len(sample_list))
            print("sample_list" + str(sample_list))
            current_train_set = [my_data_set[example] for example in sample_list]
            classifier = DecisionTreeClassifier()
            start = time.time()
            classifier.train(current_train_set, label, label_types)
            end = time.time()
            print("in " + str(i) + " iterations: time = " + str((end - start)) + " seconds")
            predictions = classifier.fit(my_data_set, label, label_types)
            correct_predictions = self._compare_list(predictions, [example[-1] for example in my_data_set])

            accuracy = correct_predictions.count(1) / len(my_data_set)
            print("i = " + str(i) + " accuracy = " + str(accuracy))
            error = sum([x * y for x, y in zip(weights, correct_predictions)]) / record_num
            print("error = " + str(error))
            print("error > 0.5 " + str((error > 0.5)))
            if error > 0.5:
                print("no, resetting weights!")
                weights = [1 / record_num] * record_num
                my_data_set.shuffle()
            importance_of_classifier = 0.5 * math.log(((1 - error)/error), math.e)
            self._clfs.append([classifier, importance_of_classifier])
            weights = self._updating_weights(importance_of_classifier, correct_predictions, weights)

    def _compare_list(self, alist, blist):
        if len(alist) != len(blist):
            print("Error, comparing list needs the two list should be same length")
            return None
        result = []
        for i in range(len(alist)):
            if alist[i] == blist[i]:
                result.append(1)
            else:
                result.append(0)
        return result

    def _updating_weights(self, importance, correct_predictions, weights):
        result = []
        for i in range(0, len(weights)):
            if correct_predictions[i] == 1:
                w = weights[i] * math.pow(math.e, -1 * importance)
            else:
                w = weights[i] * math.pow(math.e, 1 * importance)
            result.append(w)
        sumw = sum(result)
        return [example / sumw for example in result]

    def fit_one(self, x, all_class, label, label_types):
        return self._fit(x, all_class, label, label_types)

    def fit(self, X, all_class, label, label_types):
        result = list()
        for item in X:
            result.append(self.fit_one(item, all_class, label, label_types))
        return result

    def _fit(self, x, all_class, label, label_types):
        result = None
        max_importance = -1
        for item in all_class:
            importance = 0.0
            for i in range(0, len(self._clfs)):
                prediction = self._clfs[i][0].fit_one(x, label, label_types)
                if prediction == item:
                    importance += self._clfs[i][1]
            if max_importance <= importance:
                max_importance = importance
                result = item
        return result


def test_DecisionTreeClassifier():
    clf = DecisionTreeClassifier()
    abalone_data = AdultData('adult.csv')
    # test case 1
    print("test case 1 start")
    print(clf._majority_vost([1, 1, 2, 2, 2, 3, 2]))
    print("test case 1 end")
    print("\n\n")

    # test case 2
    print("test case 2 start")
    data_set1 = [['M', 'F', '1'],
                ['M', 'F', '1'],
                ['P', 'T', '0'],
                ['P', 'T', '0'],
                ['M', 'T', '0']
                ]  # entropy is 0.9709505944546686

    data_set2 = [['M', 'F', '1'],
                 ['M', 'F', '1'],
                 ['P', 'T', '1'],
                 ['P', 'T', '0'],
                 ['M', 'T', '0']
                 ] # entropy is 0.9709505944546686

    data_set3 = [['M', 'F', '1'],
                 ['M', 'F', '1'],
                 ['P', 'T', '1'],
                 ['P', 'T', '0'],
                 ['M', 'T', '0'],
                 ['M', 'T', '0']
                 ]
    data_set4 = [['M', 'F', '1'],
                 ['M', 'F', '1'],
                 ['P', 'T', '1'],
                 ['P', 'T', '1'],
                 ['M', 'T', '1'],
                 ]
    print(clf._calc_entropy(data_set1))
    print(clf._calc_entropy(data_set2))
    print(clf._calc_entropy(data_set3))
    print(clf._calc_entropy(data_set4))
    print("test case 2 end")
    print("\n\n")

    # test case 3
    print("test case 3 start")
    print(clf._gain_best_split(data_set1))
    print(clf._gain_best_split(data_set2))
    print(clf._gain_best_split(data_set3))
    print(clf._gain_best_split(data_set4))
    print("test case 3 end")
    print("\n\n")

    # print(trainer._choose_best_split_attr(abalone_data.divide_holdout()[0]))

    # test case 4
    print("test case 4 start")
    case5_data_set = [['M', 'F', '1'],
                 ['M', 'F', '1'],
                 ['P', 'T', '1'],
                 ['P', 'T', '1'],
                 ['M', 'T', '1'],
                 ]
    print(clf._find_record_with_value(1, case5_data_set, 'T'))
    print(clf._find_record_with_value(1, case5_data_set, 'F'))
    print(clf._find_record_with_value(1, case5_data_set, 'X'))
    print("test case 4 end")
    print("\n\n")

    # test case 5
    print("test case 5 start")
    case6_data_set = [['1', 'F', '1'],
                      ['2', 'F', '1'],
                      ['3', 'T', '1'],
                      ['4', 'T', '1'],
                      ['5', 'T', '1'],
                      ]
    p = clf._split_data(0, case6_data_set, 0.5)
    print(p[0])
    print(p[1])

    p = clf._split_data(0, case6_data_set, 1.5)
    print(p[0])
    print(p[1])

    p = clf._split_data(0, case6_data_set, 2.5)
    print(p[0])
    print(p[1])

    p = clf._split_data(0, case6_data_set, 3.5)
    print(p[0])
    print(p[1])

    p = clf._split_data(0, case6_data_set, 4.5)
    print(p[0])
    print(p[1])

    p = clf._split_data(0, case6_data_set, 5.5)
    print(p[0])
    print(p[1])
    print("test case 5 end")
    print("\n\n")

    # test case 7 test _remove_feature()
    print("test case 6 start")
    print(clf._remove_feature(1, case6_data_set))
    print("test case 6 end")
    print("\n\n")

    clf.train_holdout(abalone_data)


def test_AdaBoostClassifier_binsearch():
    clf = AdaBoostClassifier()
    a = [0, 9, 10, 19, 20, 21]
    print(clf._binsearch(a, 11, 0, 5)) # correct = 2
    print(clf._binsearch(a, 0, 0, 5)) # correct = 0
    print(clf._binsearch(a, 21, 0, 5)) # corrct = 4
    print(clf._binsearch(a, -18, 0, 5)) # corret = -1
    print(clf._binsearch(a, 22, 0, 5)) # correct = -1
    print(clf._binsearch(a, 255, 0, 5))# corrct = -1


def test_Adaboost_sampleWithWeights():
    clf = AdaBoostClassifier()
    a = [0.2, 0.2, 0.2, 0.2, 0.2]
    print(clf._sample_with_weights(a))
    b = [0.1, 0.1, 0.1, 0.1, 0.6]
    print(clf._sample_with_weights(b))


if __name__ == '__main__':
    # test_decision_tree_classifier()
    # test_AdaBoostClassifier_binsearch()
    test_Adaboost_sampleWithWeights()
