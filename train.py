import sys


from adaboost import *
from decision_tree import *
# classification is +1 for english , -1 for Dutch


MAX_TREE_DEPTH = 8
DESCISION_STUMPS = 8

vowels = ['A', 'E', 'I', 'O', 'U']
dutch_diphtongs = ['ae', 'ei', 'au', 'ai', 'eu', 'ie', 'oe', 'ou',
                   'ui', 'aai', 'oe', 'ooi', 'eeu', 'ieu']
frequent_dutch_words = ['ik', 'je', 'het', 'de', 'is', 'dat', 'een', 'niet', 'en', 'wat']
frequent_english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']
stop_words_english = {"isn't", 'theirs', 'ain', 'herself', 'they', "shouldn't", 'mustn',
                      'but', 'those', 'now', 'more', "wasn't", 'some', 'can', 'most',
                      'wouldn', 'doesn', 'no', "couldn't", 'below', 'few', 'at', 'other',
                      'being', 're', 'don', 'him', 'whom', 'her', 'do', "hasn't", 'shouldn',
                      'he', "don't", 'their', 'yours', 'about', 'yourselves', 'during',
                      'which', 'when', 'is', 'ourselves', 'd', "mightn't", "hadn't", 'doing',
                      'o', 'over', 'our', 'with', "it's", 'ours', "mustn't", 'shan', "wouldn't",
                      'his', 'through', 'me', 'hers', 'you', 'too', 'then', 'were', 'again',
                      'once', "she's", 'just', 'while', "aren't", 'down', "didn't", 'mightn',
                      'have', "haven't", 'why', "weren't", 'my', 'had', 'and', 'i', 'such',
                      'weren', 'in', 'under', 'not', 'll', 'same', 'she', 'does', 'himself',
                      'against', 'been', 'hasn', 'won', 'above', 'did', 'its', "should've",
                      'this', 'myself', 'am', 'out', 'hadn', 'we', 'are', 'your', 'has',
                      'the', 'than', 'if', 'will', 'them', 'having', 'there', 'what',
                      'couldn', 'needn', 'that', 's', 'as', 'm', 'between', 'because',
                      'itself', "needn't", "shan't", 'by', 'up', "won't", 'aren', "doesn't",
                      'isn', 'wasn', 'own', 'these', 'a', 'be', 'each', 'for', 'how', 'an',
                      'here', 'it', "you've", 'any', 'both', 'or', "you're", 've', 'all',
                      'yourself', "you'll", 'off', 'into', 'haven', 'only', 'on', "that'll",
                      'themselves', 'until', 'so', 'of', 'after', 'ma', "you'd", 'to', 'from',
                      'didn', 'who', 'where', 'y', 'further', 'nor', 'should', 't', 'before',
                      'very', 'was'}
stop_words_dutch = {'tegen', 'hun', 'een', 'dit', 'worden', 'geweest', 'waren', 'dan', 'naar',
                    'doch', 'wordt', 'zonder', 'uit', 'om', 'is', 'hij', 'dus', 'alles', 'over',
                    'dat', 'als', 'door', 'al', 'me', 'men', 'wie', 'bij', 'mijn', 'het', 'had',
                    'heeft', 'in', 'er', 'geen', 'die', 'je', 'van', 'zelf', 'zou', 'hier', 'onder',
                    'nu', 'hem', 'heb', 'de', 'nog', 'meer', 'reeds', 'moet', 'toen', 'tot', 'zich',
                    'der', 'hebben', 'voor', 'niet', 'toch', 'met', 'ge', 'haar', 'hoe', 'uw', 'zijn',
                    'wil', 'eens', 'iemand', 'kan', 'ons', 'ben', 'altijd', 'zij', 'aan', 'maar', 'op',
                    'kon', 'doen', 'te', 'u', 'niets', 'andere', 'mij', 'ik', 'zo', 'veel', 'werd',
                    'deze', 'en', 'wat', 'kunnen', 'omdat', 'wezen', 'of', 'daar', 'want', 'ook', 'na',
                    'was', 'zal', 'ze', 'ja', 'iets'}


def has_repeating_vowels(elements):
    # Dutch language has high probability of repeating vowels
    for element in elements:
        element = element.upper()
        if 'AA' in element or 'EE' in element or 'II' in element or 'OO' in element or 'UU' in element:
            return True
    return False


def has_ij(elements):
    # Dutch language has high probability of ij in words
    for element in elements:
        element = element.upper()
        if 'IJ' in element:
            return True
    return False


def has_dutch_frequent_words(elements):
    for element in elements:
        element = element.lower()
        if element in frequent_dutch_words:
            return True
    return False


def has_english_frequent_words(elements):
    for element in elements:
        element = element.lower()
        if element in frequent_english_words:
            return True
    return False


def has_dutch_diphtongs(elements):
    for element in elements:
        element = element.lower()
        for diphtong in dutch_diphtongs:
            if diphtong in element:
                return True
    return False


def has_english_stopword(elements):
    for element in elements:
        if element in stop_words_english:
            return True
    return False


def has_dutch_stopword(elements):
    for element in elements:
        if element in stop_words_dutch:
            return True
    return False


def feature_selection_and_training_dataframe_creation(data):
    """
    selecting the features based on input data
    :param data: ip data
    :return: created dataframe
    """

    # feature matrix to create the training dataframe
    feature_matrix = []
    # column names
    column_names = ['vow_rpt', 'ij', 'dutch_freq', 'eng_freq', 'dutch_dipht', 'dutch_stop', 'eng_stop']

    for i in range(0, len(data)):
        # # print(str(data[i][1] + "  ( language is :  " + data[i][0] + " ) "))
        sentence_features = []
        sentence_list = data[i][1].split(' ')
        label = data[i][0]
        # print("the elements list is : " + str(sentence_list))
        sentence_features.append(has_repeating_vowels(sentence_list))
        sentence_features.append(has_ij(sentence_list))
        sentence_features.append(has_dutch_frequent_words(sentence_list))
        sentence_features.append(has_english_frequent_words(sentence_list))
        sentence_features.append(has_dutch_diphtongs(sentence_list))
        sentence_features.append(has_dutch_stopword(sentence_list))
        sentence_features.append(has_english_stopword(sentence_list))
        if label == 'en':
            sentence_features.append(label)
        else:
            sentence_features.append(label)
        # print("features and label for this sentence are : ")
        # print(str(sentence_features) + "\n")
        feature_matrix.append(sentence_features)

    training_dataframe = pd.DataFrame(feature_matrix, columns=(column_names + ['output']))
    # print("the final created training dataframe")
    # print(training_dataframe)

    return training_dataframe


if __name__ == '__main__':
    # check if correct number of arguments are provided
    if len(sys.argv) != 4:
        # print("Usage: train <examples> <hypothesisOut> <learning-type>")
        sys.exit()

    # read in command line arguments
    examples_file = sys.argv[1]
    hypothesisOut_file = sys.argv[2]
    learning_type = sys.argv[3]

    data = []
    with open(examples_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove the last '|' character and split the line into the language code and text
            lang, text = line.rstrip().split('|', 1)
            # # print(str(lang) + " " + str(text))
            # Add the language code and text as a tuple to the data list
            data.append([lang[:3], text.strip()])

    # make the features and store it in a dataframe to do the training
    training_df = feature_selection_and_training_dataframe_creation(data)
    attributes = training_df.columns.tolist()[:-1]
    # perform training based on learning_type
    if learning_type == "dt":
        # decision tree training
        # additional parameters like max tree depth can be controlled using constants in the code
        # first we will train using our created dataframe
        # learning_decision_tree(examples, attributes, parent_examples, depth, max_depth)
        # print("\n creating a decision tree : ")
        dt_root = learning_decision_tree(training_df, attributes, None, 0, MAX_TREE_DEPTH)

        # print("\n # printing the decision tree in level - wise order !!!")
        BFS(dt_root)

        # print("serializing the decision tree !! ")
        # assuming you have the root of your decision tree saved in a variable called 'tree_root'
        with open(hypothesisOut_file, 'wb') as f:
            pickle.dump(dt_root, f)
        # print("serialized !!")

    elif learning_type == "ada":
        # ada boost training
        # additional parameters like number of stumps can be controlled using constants in the code
        # print("training using the adaboost algorithm !!!")
        h, z = ADABOOST(training_df, DESCISION_STUMPS, attributes)
        # print("the hypothesis are : " + str(h))
        # print("the weights are : " + str(z))

        # print("serializing the ensemble !!!")
        with open(hypothesisOut_file, 'wb') as f:
            pickle.dump((h, z), f)
        # print("serialized !!")
    else:
        # print("Invalid learning-type. Choose either 'dt' or 'ada'")
        sys.exit()
