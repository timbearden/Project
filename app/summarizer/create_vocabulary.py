from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from summarize import clean_text
import pandas as pd
import pickle

# mongo_client = MongoClient()
# db = mongo_client.g_project_data
# coll = db.test_data
#
# summary_list = []
# article_list = []
# for doc in list(coll.find()):
#     if doc['full_text'] != ' ':
#         summary_list.append(doc['summary'])
#         article_list.append(doc['full_text'])
#
# for i in xrange(len(article_list)):
#     text = ''
#     for article in article_list[i]:
#         text += article
#     article_list[i] = text
#
# summary_test = [summary_list[i] for i in xrange(len(summary_list)) if article_list[i] != '' and sent_tokenize(article_list[i]) > 10]
# article_test = [article for article in article_list if article != '' and sent_tokenize(article) > 10]
#
# mongo_client.close()

articles = pd.read_csv('../data/articles.csv')
article_list = articles.body.copy()
#
# lem = WordNetLemmatizer()
#
# for i in xrange(len(article_list)):
#     article = ' '.join([lem.lemmatize(lem.lemmatize(word, pos ='v')) for word in article_list[i].split()])
#     article = ' '.join([lem.lemmatize(lem.lemmatize(word)) for word in article.split()])
#     article_list[i] = article

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(article_list)

idf = tfidf.idf_
vocab = tfidf.vocabulary_

def picklize(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f)

picklize(idf, 'idf')
picklize(vocab, 'vocab')
