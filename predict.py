import sys

from train import *


def dfs(root, vals):
    if isinstance(root, LeafNode):
        return root.classification

    split_feature = root.attribute
    if vals[split_feature] == True:
        return dfs(root.left, vals)
    else:
        return dfs(root.right, vals)


def classify_sentence_dt(sentence_num, root, pred_df):
    decision = dfs(root, pred_df.iloc[sentence_num])
    return decision


def create(data):
    # feature matrix to create the training dataframe
    feature_matrix = []
    # column names
    column_names = ['vow_rpt', 'ij', 'dutch_freq', 'eng_freq', 'dutch_dipht', 'dutch_stop', 'eng_stop']

    for i in range(0, len(data)):
        # print(str(data[i][1] + "  ( language is :  " + data[i][0] + " ) "))
        sentence_features = []
        sentence_list = data[i].split(' ')
        # print("the elements list is : " + str(sentence_list))
        sentence_features.append(has_repeating_vowels(sentence_list))
        sentence_features.append(has_ij(sentence_list))
        sentence_features.append(has_dutch_frequent_words(sentence_list))
        sentence_features.append(has_english_frequent_words(sentence_list))
        sentence_features.append(has_dutch_diphtongs(sentence_list))
        sentence_features.append(has_dutch_stopword(sentence_list))
        sentence_features.append(has_english_stopword(sentence_list))
        feature_matrix.append(sentence_features)

    testing_dataframe = pd.DataFrame(feature_matrix, columns=column_names)
    # print("the final created training dataframe")
    # print(training_dataframe)

    return testing_dataframe


def classify_sentence_ada(index, h, z, prediction_df):
    final_ans = 0
    for i in range(0, len(h)):
        decision = 1 if dfs(h[i], prediction_df.iloc[index]) == 'en' else -1
        final_ans += (z[i] * decision)

    # print("final ans : " + str(final_ans))
    return 'en' if final_ans > 0 else 'nl'


if __name__ == "__main__":
    # Parse command line arguments
    hypothesis_file = sys.argv[1]
    testing_file = sys.argv[2]

    # first we will convert the test file content to a testing dataframe
    testing_data = []
    with open(testing_file, 'r', encoding='utf-8') as f:
        for line in f:
            testing_data.append(line.strip())

    # make frame
    prediction_df = create(testing_data)
    # print("the prediction dataframe is : ")
    # print(prediction_df)

    with open(hypothesis_file, 'rb') as f:
        hypothesis = pickle.load(f)

    if isinstance(hypothesis, Node):
        # print("this is a node !!!! , decision tree classifier will be used !!!!")
        dt_root = hypothesis
        # print("deserialized checking !!!")
        # BFS(dt_root)
        # Classify each sentence in input file using the trained model
        # print("classification results ... ")
        index = 0
        with open(testing_file, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                # print(sentence)
                label = classify_sentence_dt(index, dt_root, prediction_df)
                print(label)
                index += 1

    elif isinstance(hypothesis, tuple):
        # print("The hypothesis is adaboost !!!!")
        with open(hypothesis_file, 'rb') as f:
            h, z = pickle.load(f)
        # print("deserialized !!")
        # print("the hypothesis are : " + str(h))
        # print("the weights are : " + str(z))
        index = 0
        with open(testing_file, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                # print(sentence)
                label = classify_sentence_ada(index, h, z, prediction_df)
                print(label)
                index += 1
