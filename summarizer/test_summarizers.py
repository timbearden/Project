from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import sent_tokenize, word_tokenize
from Rouge import rouge_score
from summarize import *
import pickle


def get_summaries_and_articles(coll):
    summary_list = []
    article_list = []

    for doc in list(coll.find()):
        if doc['full_text'] != ' ':
            summary_list.append(doc['summary'])
            article_list.append(doc['full_text'])

    for i in xrange(len(article_list)):
        text = ''
        for article in article_list[i]:
            text += article
        article_list[i] = text

    summary_test = [summary_list[i] for i in xrange(len(summary_list)) if article_list[i] != '' and sent_tokenize(article_list[i]) > 10]
    article_test = [article for article in article_list if article != '' and sent_tokenize(article) > 10]

    return summary_test, article_test


def make_article_vectors(article_list, vocab, normalize = False):
    article_vectors = []
    sentence_list = []
    for article in article_list:
        sentences = np.array(sent_tokenize(article))
        sentence_list.append(sentences)
        if normalize:
            counts = CountVectorizer(stop_words='english',vocabulary=vocab, normalize=True)
        else:
            counts = CountVectorizer(stop_words='english',vocabulary=vocab)
        article_count_vector = get_vector(counts, [article])[0]
        article_vectors.append(article_count_vector)
    return np.array(sentence_list), np.array(article_vectors)



if __name__ == '__main__':
    mongo_client = MongoClient()
    db = mongo_client.g_project_data
    coll = db.test_data

    summary_list, article_list = get_summaries_and_articles(coll)

    mongo_client.close()

    idf = unpickle('idf')
    vocab = unpickle('vocab')

    sentence_list, article_vectors = make_article_vectors(article_list, vocab)

    random_rouge = []
    for sentences, vector, summary in zip(sentence_list, article_vectors, summary_list):
        scores = random_baseline(sentences)
        important_sentences = get_important_sentences(scores, sentences)
        auto_summary = make_summary(important_sentences)
        rouge = rouge_score(auto_summary, summary)
        random_rouge.append(rouge)

    significance_rouge = []
    for sentences, vector, summary in zip(sentence_list, article_vectors, summary_list):
        scores = significance_factor(vocab, vector, sentences)
        important_sentences = get_important_sentences(scores, sentences)
        auto_summary = make_summary(important_sentences)
        rouge = rouge_score(auto_summary, summary)
        significance_rouge.append(rouge)

    tfidf_rouge = []
    for sentences, vector, summary in zip(sentence_list, article_vectors, summary_list):
        scores = tfidf_corpus(vector, sentences, idf, vocab)
        important_sentences = get_important_sentences(scores, sentences)
        auto_summary = make_summary(important_sentences)
        rouge = rouge_score(auto_summary, summary)
        tfidf_rouge.append(rouge)

    print "Random Rouge: ", str(np.mean(random_rouge))
    print "Significance Rouge: ", str(np.mean(significance_rouge))
    print "TfIdf Rouge: ", str(np.mean(tfidf_rouge))
