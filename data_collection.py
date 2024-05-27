import random

import nltk
import pandas as pd
import wikipedia
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
langs = ['en', 'nl']
# num_sentences = 5

en_sentences = set()
nl_sentences = set()


for lang in langs:
    print("for language : " + str(lang))
    wikipedia.set_lang(lang)
    titles = wikipedia.random(10000)
    for title in titles:
        try:
            summary = wikipedia.summary(title)
            sentences = sent_tokenize(summary)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                if len(tokens) == 15:
                    if lang == 'en' and sentence not in en_sentences:
                        en_sentences.add(sentence)
                    elif lang == 'nl' and sentence not in nl_sentences:
                        nl_sentences.add(sentence)
                elif len(tokens) > 15:
                    random_indexes = random.sample(range(len(tokens)), 15)
                    random_indexes.sort()
                    sentence = " ".join([tokens[i] for i in random_indexes])
                    if lang == 'en' and sentence not in en_sentences:
                        en_sentences.add(sentence)
                    elif lang == 'nl' and sentence not in nl_sentences:
                        nl_sentences.add(sentence)
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
            print("Exception: ")

en_data = [[sentence, len(word_tokenize(sentence)), 'en'] for sentence in en_sentences]
nl_data = [[sentence, len(word_tokenize(sentence)), 'nl'] for sentence in nl_sentences]

data = en_data + nl_data
df = pd.DataFrame(data, columns=['sentence', 'length', 'lang'])
print(df)
print()
print("serial - deserialize")
df.to_pickle('dataframe.pkl')
df = pd.read_pickle('dataframe.pkl')

# print the deserialized DataFrame
print(df)
