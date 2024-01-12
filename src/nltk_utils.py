from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

stoplist = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def delete_stopwords(article: str) -> str:
    """perform the stopwords elimination"""
    return [word.lower() for word in article if word.lower() not in stoplist]

def get_tokens(text: str) -> list:
    return word_tokenize(text)

def get_sentences(text: str) -> list:
    return sent_tokenize(text)

def perform_stemming(text: list) -> list:
    return [ps.stem(word) for word in text]

# perform the lemmatization of the article
def lemmatize_article(article: list) -> list:
    return [lemmatizer.lemmatize(word) for word in article]