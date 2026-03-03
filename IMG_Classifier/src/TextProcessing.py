from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer

def processing_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('spanish'))
    stemmer = PorterStemmer()

    tokens = tokenizer.tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)