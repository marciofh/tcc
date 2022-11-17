import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('rslp')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("pt_core_news_sm")

# ================================= DATASET =================================
df = pd.read_csv("rumor-election-brazil-2018.csv",delimiter=";")
df = df.dropna()
# print(df.head(5))
stop = stopwords.words('portuguese')
# print(stop)
list_stop_words = ['em','sao','ao','de','da','do','para','c','kg','un',
              'ml','pct','und','das','no','ou','pc','gr','pt','cm',
              'vd','com','sem','gfa','jg','la','1','2','3','4','5',
              '6','7','8','9','0','a','b','c','d','e','lt','f','g',
              'h','i','j','k','l','m','n','o','p','q','r','s','t',
              'u','v','x','w','y','z']

#================================== PRÉ PROCESSAMENTO =================================== Não seria interessante preservar os números?
def processa_texto(text):
    text = str(text).lower() 
    text = re.sub('\[.*?\]','', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+','', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df["texto"] = df["texto"].apply(processa_texto)

# LEMATIZAÇÃO
def lemmatize_words(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lem = ' '.join([wnl.lemmatize(word) for word in text.split()])
    return lem

df['texto'] = df['texto'].apply(lemmatize_words)

#TOKENIZAÇÃO
# print(nltk.word_tokenize(df['texto'][0]))
cvt = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words=list_stop_words) #acentos, minúsculas, stopwords
tokens = cvt.fit_transform(df['texto']) # Transformação em vetor binário
tokens = tokens.toarray()
# print(tokens)
print(cvt.get_feature_names())
print(len(cvt.get_feature_names())) #1864 palavras

#TF-IDF
tf_transformer = TfidfTransformer(use_idf=True) # use_idf: Opção de normalização tf-idf
td_idf = tf_transformer.fit_transform(tokens) # Transformação com normalização tf-idf
td_idf = td_idf.toarray()
# print(td_idf)

#WORD CLOUD
def gera_wordcloud(df):
    text = " ".join(df['texto'])
    wordcloud = WordCloud(
        width = 3000, 
        height = 2000, 
        background_color = 'black', 
        min_font_size = 20,
        stopwords = set(nltk.corpus.stopwords.words("portuguese"))
    ).generate(text)
    plt.figure(figsize = (40, 30)) 
    plt.imshow(wordcloud,interpolation='bilinear') 
    plt.axis("off")
    plt.show() 

# gera_wordcloud(df)

# =================================== MODELO ===================================
y = df['rotulo']
X = df['texto']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=53)

#================================== CLASSIFICAÇÃO ===================================
# Support Vector Machine (SVM)
# Naive Bayes Classifier
# Árvore de Decisão
# Rede Neural (LSTM)