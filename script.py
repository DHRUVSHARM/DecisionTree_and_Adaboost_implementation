import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
if __name__ == '__main__':
    nltk.download('stopwords')
    stop_words_english = set(stopwords.words('english'))
    stop_words_dutch = set(stopwords.words('dutch'))

    print(stop_words_english)
    print(stop_words_dutch)