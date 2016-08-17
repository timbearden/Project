from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import sent_tokenize, word_tokenize
from Rouge import rouge_score
from summarize import *
from summarizer import Summarizer
import matplotlib.pyplot as plt
import seaborn
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

    summary_test = np.unique([summary_list[i] for i in xrange(len(summary_list)) if article_list[i] != '' and article_list[i] != ' ' and len(sent_tokenize(article_list[i])) > 10])
    article_test = np.unique([article for article in article_list if article != '' and article_list[i] != ' ' and len(sent_tokenize(article)) > 10])

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

    count = CountVectorizer(vocabulary=vocab, stop_words='english')

    summarizer_multi = Summarizer(vocab=vocab, idf=idf, vectorizer=count, scoring='multi_Tfidf')
    summarizer_single = Summarizer(vocab=vocab, idf=idf, vectorizer=count, scoring='single_Tfidf')
    summarizer_sig = Summarizer(vocab=vocab, idf=idf, vectorizer=count, scoring='significance')
    summarizer_sim = Summarizer(vocab=vocab, idf=idf, vectorizer=count, scoring='similarity')
    summarizer_rand = Summarizer(vocab=vocab, idf=idf, vectorizer=count, scoring='random')

    multi_r2 = []
    multi_reduction2 = []
    single_r2 = []
    single_reduction2 = []
    sig_r2 = []
    sig_reduction2 = []
    sim_r2 = []
    sim_reduction2 = []
    rand_r2 = []
    rand_reduction2 = []
    for summary, article in zip(summary_list[53:], article_list[53:]):
        summarizer_multi.fit(article)
        summarizer_single.fit(article)
        summarizer_sig.fit(article)
        summarizer_sim.fit(article)
        summarizer_rand.fit(article)
        multi_r2.append(summarizer_multi.rouge(summary))
        single_r2.append(summarizer_single.rouge(summary))
        sig_r2.append(summarizer_sig.rouge(summary))
        sim_r2.append(summarizer_sim.rouge(summary))
        rand_r2.append(summarizer_rand.rouge(summary))
        multi_reduction2.append(summarizer_multi.reduction)
        single_reduction2.append(summarizer_single.reduction)
        sig_reduction2.append(summarizer_sig.reduction)
        sim_reduction2.append(summarizer_sim.reduction)
        rand_reduction2.append(summarizer_rand.reduction)

    plt.boxplot([multi_r, single_r, sig_r, sim_r, rand_r])
    plt.ylabel('Rouge Score')
    plt.savefig('../images/boxplot.png')

    plt.boxplot([multi_r, sig_r, rand_r])
    plt.ylabel('Rouge Score')
    plt.savefig('../images/boxplot2.png')
    #
    # sentence_list, article_vectors = make_article_vectors(article_list, vocab)
    #
    # random_rouge = []
    # for sentences, vector, summary in zip(sentence_list, article_vectors, summary_list):
    #     scores = random_baseline(sentences)
    #     important_sentences = get_important_sentences(scores, sentences)
    #     auto_summary = make_summary(important_sentences)
    #     rouge = rouge_score(auto_summary, summary)
    #     random_rouge.append(rouge)
    #
    # significance_rouge = []
    # for sentences, vector, summary in zip(sentence_list, article_vectors, summary_list):
    #     scores = significance_factor(vocab, vector, sentences)
    #     important_sentences = get_important_sentences(scores, sentences)
    #     auto_summary = make_summary(important_sentences)
    #     rouge = rouge_score(auto_summary, summary)
    #     significance_rouge.append(rouge)
    #
    # tfidf_rouge = []
    # for sentences, vector, summary in zip(sentence_list, article_vectors, summary_list):
    #     scores = tfidf_corpus(vector, sentences, idf, vocab)
    #     important_sentences = get_important_sentences(scores, sentences)
    #     auto_summary = make_summary(important_sentences)
    #     rouge = rouge_score(auto_summary, summary)
    #     tfidf_rouge.append(rouge)
    #
    # print "Random Rouge: ", str(np.mean(random_rouge))
    # print "Significance Rouge: ", str(np.mean(significance_rouge))
    # print "TfIdf Rouge: ", str(np.mean(tfidf_rouge))
