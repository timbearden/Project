from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import random
from nltk import sent_tokenize, word_tokenize
from summary_mining import get_full_article, get_summary_and_full_links
import pickle
import re
from Rouge import rouge_score


def unpickle(filename):
    '''
    INPUT: filename (string)
    OUTPUT: object

    Unpickles a specified file; either vocab dictionary or idf matrix for this script.
    '''
    with open(filename) as f:
        result = pickle.load(f)
    return result


def clean_text(text):
    new_text = re.sub(r'(Advertisement.*?\n)','',text)
    new_text = re.sub(r'(Photo.*?\n)','',new_text)

def get_vector(vectorizer, document, normalize=False):
    '''
    INPUT: vectorizer object, list/array of document(s), bool (optional)
    OUTPUT: vectorized array

    Creates a single vector if given the full news article, or creates an array of vectors
    for each sentence.
    '''
    vector = vectorizer.fit_transform(document).todense()
    vector = np.array(vector)
    if normalize:
        norm_vec = [vec/float(len(word_tokenize(doc))) for vec, doc in zip(vector, document)]
        vector = norm_vec
    vector = np.array(vector)
    return vector


def significance_factor(vocab,full_vec,sentences):
    '''
    INPUT: vocabulary dictionary, vector array, sentence array
    OUTPUT: array

    Scores each sentence based off significance; sums up document counts or tfidf for each
    word in the sentence.
    '''
    sentence_scores = []
    for sentence in sentences:
        score = np.sum([full_vec[vocab[word.lower()]] for word in sentence.split() if word.lower() in vocab.keys()])
        sentence_scores.append(score)
    return np.array(sentence_scores)


def get_sentence_cos_sims(sentence_vecs, full_vec):
    '''
    INPUT: Two vector arrays
    OUTPUT: Array

    Scores each sentence based off of cosine similarity to the entire document.
    '''
    similarities = []
    for vec in sentence_vecs:
        nan_idx = np.isnan(vec)
        vec[nan_idx] = 0
        similarity = cosine_similarity(full_vec, vec).flatten()
        similarities.append(similarity)
    similarities = np.array(similarities).flatten()
    return similarities


def tfidf_single(sentences, n1=1, n2=1):
    '''
    INPUT: Sentence array, lower n-gram (int), upper n-gram (int)
    OUTPUT: array

    Treats the entire article as a corpus of sentence documents, finds tf-idf
    scores for each sentence.
    '''
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(n1,n2))
    tfidf_mat = tfidf_vectorizer.fit_transform(sentences).todense()
    tfidf_scores = np.array(tfidf_mat.sum(axis=1).flatten())[0,:]
    return tfidf_scores


def tfidf_corpus(count_vec, sentences, idf, vocab):
    '''
    INPUT: Vector array, sentence array, idf matrix, vocab dictionary
    OUTPUT: array

    Creates tfidf scores for each sentence based off larger corpus of news articles.
    Calculates scores similarly to significance scores.
    '''
    tfidf = count_vec * idf
    tfidf_scores = []
    for sentence in sentences:
        score = np.sum([tfidf[vocab[word.lower()]] for word in sentence.split() if word.lower() in vocab.keys()])
        tfidf_scores.append(score)
    return np.array(tfidf_scores)

def get_important_sentences(importance_ratings, sentences, num_sentences=None):
    '''
    INPUT: scoring array, sentence array, number of sentences (optional)
    OUTPUT: array of sentences

    Creates an array of important sentences ranked by the given scoring metric.
    '''
    sort_idx = np.argsort(importance_ratings)[::-1]
    if num_sentences==None:
        num_sentences = min(max(len(sentences)*0.10, 5), 10)
    important_sentence_idx = sort_idx[:num_sentences]
    sentence_idx = np.sort(important_sentence_idx)
    summary_array = sentences[sentence_idx]
    return summary_array

def just_topic_sentences(sentences):
    '''
    INPUT: Sentence array
    OUTPUT: Topic sentence array

    Creates an array of the topic sentences in an article for a baseline summary.
    '''
    topic_sentence_idx = np.array([i for i in xrange(len(sentences)) if '\n' in sentences[i]])
    topic_sentences = sentences[topic_sentence_idx]
    return topic_sentences

def first_n_sentences(sentences, n):
    '''
    INPUT: Sentence array, int
    OUTPUT: Array

    Creates an array of the first n sentences of an article for a baseline summary.
    '''
    first_sentences = sentences[:n]
    return first_sentences


def naive_bayes(sentences, prior = 0.10):
    '''
    INPUT:
    OUTPUT:
    '''
    pass


def hidden_markov(sentences):
    '''
    INPUT:
    OUTPUT:
    '''
    pass


def log_linear(sentences):
    '''
    INPUT:
    OUTPUT:
    '''
    pass


def random_baseline(sentences):
    '''
    INPUT: Sentence array
    OUTPUT: array

    Creates an array of sentences randomly picked from the article, for the
    simplest baseline.
    '''
    rand_scores = [random() for x in xrange(len(sentences))]
    return rand_scores


def make_summary(summary_array):
    '''
    INPUT: array
    OUTPUT: string

    Takes an array of sentences for the summary and strings them together into a readable summary.
    '''
    summary = ""
    for sentence in summary_array:
        summary += sentence + ' '
    return summary.replace('\n\n','\n')


def print_nice_summary(title, summary, test_summary):
    '''
    INPUT: string, string, string
    OUTPUT: string

    Prints out the given summary and Rouge score in a readable manner.
    '''
    print title, 'Summary:\n', summary, '\n', 'Rouge: ', str(rouge_score(summary, test_summary)), '\n'


if __name__ == '__main__':
    # summary_url = 'http://www.newser.com/story/229050/strange-saga-of-sports-legends-remains-is-still-unfolding.html'
    # url = 'http://espn.go.com/espn/feature/story/_/id/17163767/heated-debate-now-lawsuit-burial-ground-jim-thorpe-remains-continues-today'

    summary_url = 'http://www.newser.com/story/229320/nj-weighs-big-fines-for-distracted-drivers-but-coffee-is-likely-safe.html'
    url = 'http://www.nj.com/traffic/index.ssf/2016/08/cops_could_soon_ticket_you_in_nj_for_drinking_a_coffee_while_driving.html#incart_most-readstarledger'

    full_text = get_full_article(url)
    test_summary = get_summary_and_full_links(summary_url)[0]

    vocab = unpickle('vocab')
    idf = unpickle('idf')

    counts = CountVectorizer(stop_words='english',vocabulary=vocab)
    tfidf = TfidfVectorizer(stop_words='english',vocabulary=vocab)

    sentences = np.array(sent_tokenize(full_text))

    article_count_vector = get_vector(counts, [full_text])[0]
    sentence_count_vector = get_vector(counts, sentences)
    article_tfidf_vector = get_vector(tfidf, [full_text])[0]
    sentence_tfidf_vector = get_vector(tfidf, sentences)

    similarities = get_sentence_cos_sims(sentence_tfidf_vector, article_tfidf_vector)
    significance = significance_factor(vocab, article_count_vector, sentences)
    tfidf_full = tfidf_corpus(article_count_vector, sentences, idf, vocab)
    random_scores = random_baseline(sentences)
    tfidf_small = tfidf_single(sentences)

    sim_summary_array = get_important_sentences(similarities, sentences, num_sentences=5)
    sig_summary_array = get_important_sentences(significance, sentences, num_sentences=5)
    tfidf_summary_array = get_important_sentences(tfidf_full, sentences, num_sentences=5)
    rand_summary_array = get_important_sentences(random_scores, sentences, num_sentences=5)
    small_tfidf_summary_array = get_important_sentences(tfidf_small, sentences, num_sentences=5)
    topic_sentences = just_topic_sentences(sentences)
    first_sentence_array = first_n_sentences(sentences, 5)

    tfidf_summary = make_summary(tfidf_summary_array)
    sim_summary = make_summary(sim_summary_array)
    sig_summary = make_summary(sig_summary_array)
    rand_summary = make_summary(rand_summary_array)
    small_tfidf_summary = make_summary(small_tfidf_summary_array)
    topic_sentence_summary = make_summary(topic_sentences)
    first_sentences_summary = make_summary(first_sentence_array)

    print_nice_summary('Big TfIdf', tfidf_summary, test_summary)
    print_nice_summary('Similarity', sim_summary, test_summary)
    print_nice_summary('Significance', sig_summary, test_summary)
    print_nice_summary('Small TfIdf', small_tfidf_summary, test_summary)
    print_nice_summary('Topic Sentences', topic_sentence_summary, test_summary)
    print_nice_summary('First Sentences', first_sentences_summary, test_summary)
    print "Random Rouge: ", str(rouge_score(rand_summary, test_summary))
