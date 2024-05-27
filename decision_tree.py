"""
author : Dhruv Sharma
ds7042@rit.edu
note input file should be named  input.txt in the same folder as the project
implementation of a decision tree for classification
Not complete used for assignment 3P
"""
from typing import Any

import numpy as np
import pandas as pd
import math
import queue
import pickle


class Node:
    def __init__(self, attribute=None, output_label_mapping=None, left=None, right=None):
        if output_label_mapping is None:
            output_label_mapping = {}
        self.attribute = attribute
        self.output_label_mapping = output_label_mapping
        self.left = left
        self.right = right

    def __str__(self):
        return f"****************************\nNode:\n attribute={self.attribute}\n " \
               f"output_label_mapping={self.output_label_mapping}\n " \
               f"*****************************\n"

    def is_leaf_node(self):
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, maximal_depth=None, root=None):
        self.maximal_depth = maximal_depth
        self.root = root


class LeafNode:
    def __init__(self, classification, mapping):
        self.classification = classification
        self.mapping = mapping
        self.left = None
        self.right = None

    def __str__(self):
        return f"****************************\nLeafNode:\n classification={self.classification}\n" \
               f"mapping={self.mapping}\n" \
               f"*****************************\n"


def plurality_value(exs) -> Any | None:
    """
    to get the maximal occurring label in data for classification
    :param exs: i/p
    :return:
    """
    if exs is None:
        return None
    common_label = exs['output'].mode()[0]
    # # print(common_label)
    return common_label


def binary_entropy(q):
    if q == 0 or q == 1:
        return 0.0
    else:
        return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


def get_maximal_info_gain(attributes, examples, distinct_value_counts):
    # print("finding maximal information gain and deciding on a split attribute ")
    # print("********************************************************************")
    # print("distinct value counts are : " + str(distinct_value_counts))
    p = distinct_value_counts['en']
    q = distinct_value_counts['nl']
    # calculating entropy of the output variable
    output_entropy = binary_entropy(p / (p + q))
    maximal_gain = -1
    best_split_attribute = None

    # here left is the df of rows where the attr value is true
    # and right is the opposite, ie; false
    for attr in attributes:
        left = examples[examples[attr] == True]['output'].value_counts().reindex(['en', 'nl'], fill_value=0)
        # # print("left is : ")
        # # print(str(left))
        pl = left['en']
        ql = left['nl']
        arg_left = 0 if (pl + ql) == 0 else (pl / (pl + ql))
        right = examples[examples[attr] == False]['output'].value_counts().reindex(['en', 'nl'], fill_value=0)
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

    # print("********************************************************************")
    return best_split_attribute


def learning_decision_tree(examples, attributes, parent_examples, depth, max_depth):
    # # print(root)
    distinct_value_counts = examples['output'].value_counts().reindex(['en', 'nl'], fill_value=0)
    # print("the attributes are : " + str(attributes))
    # # print("distinct value counts")
    # # print(distinct_value_counts['en'])

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
        best_split_attribute = get_maximal_info_gain(attributes, examples, distinct_value_counts)
        # here all attributes have value true or false only
        True_df = examples[examples[best_split_attribute] == True]
        # print("the true one : ")
        # print(True_df)
        False_df = examples[examples[best_split_attribute] == False]
        # print("the false one : ")
        # print(False_df)
        # print("best split attribute : " + best_split_attribute)
        new_attributes = [attr for attr in attributes if attr != best_split_attribute]
        # print("new attr : " + str(new_attributes))
        root.left = learning_decision_tree(True_df, new_attributes, examples, depth + 1, max_depth)
        root.right = learning_decision_tree(False_df, new_attributes, examples, depth + 1, max_depth)
        root.attribute = best_split_attribute
    return root


def BFS(decision_tree_root):
    # Create a new empty queue
    nodes = queue.Queue()
    nodes.put(decision_tree_root)

    level = 0

    while not nodes.empty():
        level_length = nodes.qsize()
        # print("************************************")
        # print("nodes at level : " + str(level))
        while level_length > 0:
            frontier = nodes.get()
            # print("node is as : ")
            # print(frontier)
            if frontier.left is not None:
                nodes.put(frontier.left)
            if frontier.right is not None:
                nodes.put(frontier.right)
            level_length -= 1
        # print("************************************\n")
        level += 1


def createDecisionTree(df):
    """
    we will create a decision tree and return the root of the tree
    :param df: input example data
    :return:
    """

    # print("the attributes are : ")
    # print(str(df.columns.tolist()))
    attributes = df.columns.tolist()[:-1]
    examples = df
    # print("the examples are : ")
    # print(examples)

    # print("\n creating a decision tree : ")
    decision_tree_root = learning_decision_tree(examples, attributes, None, 0, 2)

    # print("\n # printing the decision tree in level - wise order !!!")
    BFS(decision_tree_root)

    # print("serializing the decision tree !! ")
    # assuming you have the root of your decision tree saved in a variable called 'tree_root'
    with open('decision_tree.pickle', 'wb') as f:
        pickle.dump(decision_tree_root, f)
    # print("serialized !!")
    # print("now deserializing !!!")
    # load the serialized decision tree from the pickle file
    with open('decision_tree.pickle', 'rb') as f:
        tree_root = pickle.load(f)
    # print("deserialized checking !!!")
    # print("\n # printing the decision tree in level - wise order !!!")
    BFS(tree_root)


if __name__ == '__main__':
    data = np.loadtxt('input.txt', delimiter=' ', dtype='object')

    # # print("the data is : ")
    # # print(data)

    # # print(len(data[0]))
    # we have a 200 X 9 matrix given to us
    '''
    count = 0
    for j in range(0, 9):
        # print(data[0][j] + " ", end='')
        if data[0][j] == True:
            count += 1
    # print("count is : " + str(count))
    '''
    dataf = pd.DataFrame(data, columns=['attribute 1', 'attribute 2', 'attribute 3', 'attribute 4',
                                        'attribute 5', 'attribute 6', 'attribute 7', 'attribute 8',
                                        'output'])

    # # print(df)
    createDecisionTree(dataf)
    # tree = createDecisionTree(data)
    # # print(df['attribute 1'])
