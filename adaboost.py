import numpy as np
import pandas as pd

from predict import dfs
from train import *
from decision_tree import *


def get_maximal_info_gain_adaboost(attributes, examples, distinct_value_counts):
    # # print("finding maximal information gain and deciding on a split attribute ")
    # # print("********************************************************************")
    # # print("distinct value counts are : " + str(distinct_value_counts))
    p = distinct_value_counts['en']
    q = distinct_value_counts['nl']
    # calculating entropy of the output variable
    output_entropy = binary_entropy(p / (p + q))
    maximal_gain = -1
    best_split_attribute = None

    # here left is the df of rows where the attr value is true
    # and right is the opposite, ie; false
    for attr in attributes:
        left = examples[examples[attr] == True].groupby('output')['weights'].sum().reindex(['en', 'nl'], fill_value=0)
        # # print("left is : ")
        # # print(str(left))
        pl = left['en']
        ql = left['nl']
        arg_left = 0 if (pl + ql) == 0 else (pl / (pl + ql))
        right = examples[examples[attr] == False].groupby('output')['weights'].sum().reindex(['en', 'nl'], fill_value=0)
        # # print("right is : ")
        # # print(str(right))
        pr = right['en']
        qr = right['nl']
        arg_right = 0 if (pr + qr) == 0 else (pr / (pr + qr))
        remainder = ((pl + ql) / (p + q)) * binary_entropy(arg_left) + \
                    ((pr + qr) / (p + q)) * binary_entropy(arg_right)

        info_gain = output_entropy - remainder
        if info_gain > maximal_gain:
            maximal_gain = info_gain
            best_split_attribute = attr

    # # print("********************************************************************")
    return best_split_attribute


def learning_decision_stump(examples, attributes, parent_examples, depth, max_depth):
    # # print(root)
    # distinct_value_counts = examples['output'].value_counts().reindex(['en', 'nl'], fill_value=0)
    # # print(distinct_value_counts)
    distinct_value_counts = examples.groupby('output')['weights'].sum().reindex(['en', 'nl'], fill_value=0)
    # # print("the attributes are : " + str(attributes))
    # # print("distinct value counts")
    # # print(distinct_value_counts)

    # base cases for creation of a leaf node
    if examples.empty:
        # no examples left we use the parent node
        return LeafNode(plurality_value(parent_examples), dict(distinct_value_counts))
    elif distinct_value_counts['en'] == 0 or distinct_value_counts['nl'] == 0:
        # all examples have a single classification
        return LeafNode('en', dict(distinct_value_counts)) if distinct_value_counts['nl'] == 0 \
            else LeafNode('nl', dict(distinct_value_counts))
    elif len(attributes) == 0 or depth == max_depth:
        return LeafNode(plurality_value(examples), dict(distinct_value_counts))
    else:
        # have to create an internal node
        root = Node(None, dict(distinct_value_counts), None, None)
        root.output_label_mapping = dict(distinct_value_counts)
        best_split_attribute = get_maximal_info_gain_adaboost(attributes, examples, distinct_value_counts)
        # here all attributes have value true or false only
        True_df = examples[examples[best_split_attribute] == True]
        # # print("the true one : ")
        # # print(True_df)
        False_df = examples[examples[best_split_attribute] == False]
        # # print("the false one : ")
        # # print(False_df)
        # # print("best split attribute : " + best_split_attribute)
        new_attributes = [attr for attr in attributes if attr != best_split_attribute]
        # # print("new attr : " + str(new_attributes))
        root.left = learning_decision_stump(True_df, new_attributes, examples, depth + 1, max_depth)
        root.right = learning_decision_stump(False_df, new_attributes, examples, depth + 1, max_depth)
        root.attribute = best_split_attribute
    return root


def ADABOOST(dataset, K, features):
    N = len(dataset)
    # # print("N : " + str(N))
    # # print("features are : " + str(features))
    w = np.empty(N)
    w.fill(1 / N)
    h = []
    # hypothesis weights
    z = []
    # K is the maximal decision stumps used for training
    for k in range(0, K):
        dataset['weights'] = w
        # # print("the weighted dataset used will be : ")
        # # print(dataset)
        h.append(learning_decision_stump(dataset, features, None, 0, 1))
        # # print("the stump is : ")
        # BFS(h[k])
        error = 0
        for j in range(0, N):
            if dfs(h[k], dataset.iloc[j]) != dataset['output'][j]:
                error = error + w[j]
        # print("the error is : " + str(error))
        if error > 0.5:
            # worse than random guessing , no effect of boosting
            # print("breaking")
            h.pop()
            break
        error = min(error, 1 - 0.0000001)
        for j in range(0, N):
            if dfs(h[k], dataset.iloc[j]) == dataset['output'][j]:
                w[j] = w[j] * error / (1 - error)

        w = [w[i] / np.sum(w) for i in range(0, len(w))]
        # normalization step
        z.append(np.log((1 - error) / error))

    # finally we return a list of dt stump roots with their corresponding weights
    return h, z


if __name__ == "__main__":
    pass
