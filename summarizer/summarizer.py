from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import random
from nltk import sent_tokenize, word_tokenize
from summary_mining import get_full_article, get_summary_and_full_links
import pickle
import re
from Rouge import rouge_score


class Summarizer(object):
    def __init__(self, vocab, idf, scoring=None):
        self.vocab = vocab
        self.idf = idf
        self.scoring = scoring
        self.sentence_vector = None
        self.article = None
        self.summary = None
        self.sentences = None


    def set_method(self):


    def clean_text(self, text, keep_quotes=True):
        '''
        INPUT: string
        OUTPUT: string

        Cleans up the text to eliminate sentences that are not actually in the article,
        keeps quotes in tact for when the text is tokenized by sentence.
        '''
        split_text = text.split(u'\u201c')
        if keep_quotes:
            for i in xrange(len(split_text)):
                more_split = split_text[i].split(u'\u201d')
                more_split[0] = more_split[0].replace('.','|')
                split_text[i] = u'\u201d'.join(more_split)
        new_text = u'\u201c'.join(split_text)
        new_text = re.sub(r'(Advertisement.*?\n)','',text)
        new_text = re.sub(r'(Photo.*?\n)','',new_text)
        new_text = re.sub(r'(?<=[A-Z])\.','',new_text)
        return new_text


    def get_vector(self, vectorizer, document, normalize=False):
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


    def significance_factor(self,vocab,full_vec,sentences):
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


    def get_sentence_cos_sims(self, sentence_vecs, full_vec):
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


    def tfidf_single(self, sentences, n1=1, n2=1):
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


    def tfidf_corpus(self, count_vec, sentences, idf, vocab):
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

    def get_important_sentences(self, importance_ratings, sentences, num_sentences=None):
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

    def just_topic_sentences(self, sentences):
        '''
        INPUT: Sentence array
        OUTPUT: Topic sentence array

        Creates an array of the topic sentences in an article for a baseline summary.
        '''
        topic_sentence_idx = np.array([i for i in xrange(len(sentences)) if '\n' in sentences[i]])
        topic_sentences = sentences[topic_sentence_idx]
        return topic_sentences

    def first_n_sentences(self, sentences, n):
        '''
        INPUT: Sentence array, int
        OUTPUT: Array

        Creates an array of the first n sentences of an article for a baseline summary.
        '''
        first_sentences = sentences[:n]
        return first_sentences


    def naive_bayes(self, sentences, prior = 0.10):
        '''
        INPUT:
        OUTPUT:
        '''
        pass


    def hidden_markov(self, sentences):
        '''
        INPUT:
        OUTPUT:
        '''
        pass


    def log_linear(self, sentences):
        '''
        INPUT:
        OUTPUT:
        '''
        pass


    def random_baseline(self, sentences):
        '''
        INPUT: Sentence array
        OUTPUT: array

        Creates an array of sentences randomly picked from the article, for the
        simplest baseline.
        '''
        rand_scores = [random() for x in xrange(len(sentences))]
        return rand_scores


    def make_summary(self, summary_array):
        '''
        INPUT: array
        OUTPUT: string

        Takes an array of sentences for the summary and strings them together into a readable summary.
        '''
        summary = ' '.join(summary_array)
        summary = summary.replace('\n\n','\n').replace('|','.')
        return summary
