from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import random
from nltk import sent_tokenize, word_tokenize
from summary_mining import get_full_article, get_summary_and_full_links
import pickle
from Rouge import rouge_score



def get_vecs(text, sentences, vocab, norm=True):
    full_vectorizer = CountVectorizer(stop_words='english', vocabulary=vocab)
    full_vectorizer.fit([text])
    ## Get normalized vector count of the full document for comparison
    full_vec = full_vectorizer.fit_transform([text]).todense()
    if norm:
        full_vec = full_vec/float(len(text.split()))
    full_vec = np.array(full_vec).flatten()
    ## Creating vectorizer based on document vocabulary
    # vocab = full_vectorizer.vocabulary_
    sentence_vectorizer = CountVectorizer(stop_words='english', vocabulary=vocab)
    ## Vectorizing each sentence in the document
    sentence_vecs = []
    for sentence in sentences:
        vec = sentence_vectorizer.fit_transform([sentence]).todense()
        if norm:
            vec = vec/float(len(sentence.split()))
        vec = np.array(vec).flatten()
        sentence_vecs.append(vec)
    sentence_vecs = np.array(sentence_vecs)
    return full_vec, sentence_vecs


def significance_factor(vocab,full_vec,sentences):
    sentence_scores = []
    for sentence in sentences:
        score = np.sum([full_vec[vocab[word.lower()]] for word in sentence.split() if word.lower() in vocab.keys()])
        sentence_scores.append(score)
    return np.array(sentence_scores)


def get_sentence_cos_sims(sentence_vecs, full_vec):
    similarities = []
    for vec in sentence_vecs:
        nan_idx = np.isnan(vec)
        vec[nan_idx] = 0
        similarity = cosine_similarity(full_vec, vec).flatten()
        similarities.append(similarity)
    similarities = np.array(similarities).flatten()
    return similarities


def tfidf_single(sentences):
    tfidf = TfidfVectorizer()
    tfidf_mat = tfidf.fit_transform(sentences).todense()
    tfidf_scores = np.array(tfidf_mat.sum(axis=1).flatten())[0,:]
    return tfidf_scores


def tfidf_corpus(count_vec, idf, sentences, vocab):
    # tfidf_scores = []
    # for i in xrange(len(count_vec)):
    #     tfidf = count_vec[i,:] * idf
    #     tfidf_scores.append(tfidf)
    tfidf = count_vec * idf
    tfidf_scores = []
    for sentence in sentences:
        score = np.sum([tfidf[vocab[word.lower()]] for word in sentence.split() if word.lower() in vocab.keys()])
        tfidf_scores.append(score)
    # tfidf_scores = np.sum(tfidf_scores, axis=1)
    return np.array(tfidf_scores)

def get_important_sentences(importance_ratings, sentences, num_sentences=None, topic_sentences=False):
    ##Finding the topic sentences
    topic_sentence_idx = []
    if topic_sentences:
        topic_sentence_idx = [0]
        topic_sentence_idx.extend([i for i in xrange(len(sentences)) if '\n' in sentences[i]])
    ##Finding the other important sentences
    sort_idx = np.argsort(importance_ratings)[::-1]
    if num_sentences==None:
        num_sentences = min(max(len(sentences)*0.10, 5), 10)
    sort_idx = np.array([x for x in sort_idx if x not in topic_sentence_idx])
    important_sentence_idx = sort_idx[:num_sentences]
    topic_sentence_idx.extend(important_sentence_idx)
    sentence_idx = np.sort(np.unique(topic_sentence_idx))
    summary_array = sentences[sentence_idx]
    return summary_array


def naive_bayes(sentences, prior = 0.10):
    pass


def hidden_markov(sentences):
    pass


def log_linear(sentences):
    pass


def random_baseline(sentences):
    rand_scores = [random() for x in xrange(len(sentences))]
    return rand_scores


def make_summary(summary_array):
    summary = ""
    for sentence in summary_array:
        summary += sentence + ' '
    return summary.replace('\n\n','\n')


if __name__ == '__main__':
    # summary_url = 'http://www.newser.com/story/229050/strange-saga-of-sports-legends-remains-is-still-unfolding.html'
    # url = 'http://espn.go.com/espn/feature/story/_/id/17163767/heated-debate-now-lawsuit-burial-ground-jim-thorpe-remains-continues-today'

    summary_url = 'http://www.newser.com/story/229221/its-currently-fine-for-lawyers-to-sexually-racially-harass-each-other.html'
    url = 'http://www.nytimes.com/2016/08/05/business/dealbook/sexual-harassment-ban-is-on-the-abas-docket.html?&_r=0'


    # url = 'http://www.nbcwashington.com/news/local/City-of-Fairfax-Mayor-Arrested-for-Distributing-Meth-Police-Say-389282112.html'
    full_text = get_full_article(url)
    test_summary = get_summary_and_full_links(summary_url)[0]

    with open('vocab') as f:
        vocab = pickle.load(f)

    with open('idf') as f:
        idf = pickle.load(f)

    sentences = np.array(sent_tokenize(full_text))
    full_vec, sentence_vecs = get_vecs(full_text, sentences, vocab, norm=False)


    # similarities = get_sentence_cos_sims(sentence_vecs, full_vec)
    significance = significance_factor(vocab, full_vec, sentences)
    tfidf_full = tfidf_corpus(full_vec, idf, sentences, vocab)
    random_scores = random_baseline(sentences)
    # sim_summary_array = get_important_sentences(similarities, sentences, num_sentences=5,topic_sentences=False)
    sig_summary_array = get_important_sentences(significance, sentences, num_sentences=5,topic_sentences=False)
    tfidf_summary_array = get_important_sentences(tfidf_full, sentences, num_sentences=5,topic_sentences=False)
    rand_summary_array = get_important_sentences(random_scores, sentences, num_sentences=5,topic_sentences=False)
    tfidf_summary = make_summary(tfidf_summary_array)
    # sim_summary = make_summary(sim_summary_array)
    sig_summary = make_summary(sig_summary_array)
    rand_summary = make_summary(rand_summary_array)
    print '''Random Summary:

    ''', rand_summary, '\n', 'Rouge: ', str(rouge_score(rand_summary, test_summary)), '\n'
    # print '''Similarity Summary:
    #
    # ''', sim_summary, '\n', 'Rouge: ', str(rouge_score(sim_summary, test_summary)), '\n'
    print '''Significance Summary:

    ''', sig_summary, '\n', 'Rouge: ', str(rouge_score(sig_summary, test_summary)), '\n'
    print '''TfIdf Summary:

    ''', tfidf_summary, '\n', 'Rouge: ', str(rouge_score(tfidf_summary, test_summary)), '\n'
