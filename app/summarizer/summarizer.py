from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import random
from nltk import sent_tokenize, word_tokenize
from summarizer_dev import unpickle
from summary_scraping import get_full_article, get_summary_and_full_links
import pickle
import re
from Rouge import rouge_score


class Summarizer(object):
    def __init__(self, vocab, idf=None, scoring=None, vectorizer=None):
        self.vocab = vocab
        self.idf = idf
        self.scoring = scoring
        self.vectorizer = vectorizer
        self.article_vector = None
        self.sentence_vectors = None
        self.article = None
        self.cleaned = None
        self.summary = None
        self.summary_sentences = None
        self.sentence_scores = None
        self.sentences = None
        self.reduction = None
        self.score = None

    def set_method(self):
        '''
        INPUT: string
        OUTPUT: None

        Sets the scoring method used for evaluating sentences.
        '''
        if self.scoring == 'multi_Tfidf':
            self.score = self.tfidf_corpus
        elif self.scoring == 'single_Tfidf':
            self.score = self.tfidf_single
        elif self.scoring == 'significance':
            self.score = self.significance_factor
        elif self.scoring == 'similarity':
            self.score = self.get_sentence_cos_sims
        elif self.scoring=='random':
            self.score = self.random_baseline
        else:
            print "Please pick a valid scoring metric"

    def clean_text(self):
        '''
        INPUT: string
        OUTPUT: string

        Cleans up the text to eliminate sentences that are not actually in the
        article, keeps quotes in tact for when the text is tokenized by sentence.
        '''
        split_text = self.article.split(u'\u201c')
        for i in xrange(1, len(split_text)):
            more_split = split_text[i].split(u'\u201d')
            more_split[0] = more_split[0].replace('.', '|')
            split_text[i] = u'\u201d'.join(more_split)
        new_text = u'\u201c'.join(split_text)
        new_text = re.sub(r'(Advertisement.*?\n)', '', new_text)
        new_text = re.sub(r'(Photo.*?\n)', '', new_text)
        new_text = re.sub(r'(?<=[A-Z])\.', '', new_text)
        new_text = re.sub(r'(Related.*?\n)', '', new_text)
        new_text = '\n'.join([sentence for sentence in new_text.split('\n') if '.' in sentence])
        return new_text

    def get_vector(self, document, normalize=False):
        '''
        INPUT: vectorizer object, list/array of document(s), bool (optional)
        OUTPUT: vectorized array

        Creates a single vector if given the full news article, or creates an array of vectors
        for each sentence.
        '''
        vector = self.vectorizer.fit_transform(document).todense()
        vector = np.array(vector)
        if normalize:
            norm_vec = [vec/float(len(word_tokenize(doc))) for vec, doc in zip(vector, document)]
            vector = norm_vec
        vector = np.array(vector)
        return vector

    def significance_factor(self, vectors):
        '''
        INPUT: vocabulary dictionary, vector array
        OUTPUT: array

        Scores each sentence based off significance; sums up document counts or tfidf for each
        word in the sentence.
        '''
        sentence_scores = []
        for sentence in self.sentences:
            score = np.sum([self.article_vector[self.vocab[word.lower()]]
                            for word in sentence.split()
                            if word.lower() in self.vocab.keys()])
            sentence_scores.append(score)
        return np.array(sentence_scores)

    def get_sentence_cos_sims(self, vectors):
        '''
        INPUT: Two vector arrays
        OUTPUT: Array

        Scores each sentence based off of cosine similarity to the entire document.
        '''
        similarities = []
        for vec in vectors:
            nan_idx = np.isnan(vec)
            vec[nan_idx] = 0
            similarity = cosine_similarity(self.article_vector, vec).flatten()
            similarities.append(similarity)
        similarities = np.array(similarities).flatten()
        return similarities

    def tfidf_single(self, vectors):
        '''
        INPUT: vector
        OUTPUT: array

        Treats the entire article as a corpus of sentence documents, finds tf-idf
        scores for each sentence.
        '''
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_mat = tfidf_vectorizer.fit_transform(self.sentences).todense()
        tfidf_scores = np.array(tfidf_mat.sum(axis=1).flatten())[0, :]
        return tfidf_scores

    def tfidf_corpus(self, vectors):
        '''
        INPUT: Vector array, sentence array, idf matrix, vocab dictionary
        OUTPUT: array

        Creates tfidf scores for each sentence based off larger corpus of news articles.
        Calculates scores similarly to significance scores.
        '''
        tfidf = self.article_vector * self.idf
        tfidf_scores = []
        for sentence in self.sentences:
            score = np.sum([tfidf[self.vocab[word.lower()]]
                            for word in sentence.split()
                            if word.lower() in self.vocab.keys()])
            tfidf_scores.append(score)
        return np.array(tfidf_scores)

    def get_important_sentences(self, importance_ratings):
        '''
        INPUT: scoring array
        OUTPUT: array of sentences

        Creates an array of important sentences ranked by the given scoring metric.
        '''
        sort_idx = np.argsort(importance_ratings)[::-1]
        cumulative_importance = np.cumsum(importance_ratings[sort_idx] / float(np.sum(importance_ratings)))
        top_n = np.where(cumulative_importance < 0.5)[0]
        important_sentence_idx = sort_idx[top_n]
        sentence_idx = np.sort(important_sentence_idx)
        summary_array = self.sentences[sentence_idx]
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

    def random_baseline(self, vector):
        '''
        INPUT: Sentence array
        OUTPUT: array

        Creates an array of sentences randomly picked from the article, for the
        simplest baseline.
        '''
        rand_scores = np.array([random() for x in xrange(len(self.sentences))])
        return rand_scores

    def make_summary(self, summary_array):
        '''
        INPUT: array
        OUTPUT: string

        Takes an array of sentences for the summary and strings them together
        into a readable summary.
        '''
        summary = ' '.join(summary_array)
        summary = summary.replace('\n\n', '\n').replace('|', '.')
        return summary

    def fit(self, article):
        self.article = article
        self.set_method()
        self.article_vector = self.get_vector([self.article]).flatten()
        cleaned = self.clean_text()
        self.sentences = np.array(sent_tokenize(cleaned))
        self.sentence_vectors = self.get_vector(self.sentences)
        self.sentence_scores = self.score(self.sentence_vectors)
        self.summary_sentences = self.get_important_sentences(self.sentence_scores)
        self.summary = self.make_summary(self.summary_sentences)
        self.reduction = len(self.summary_sentences) / float(len(self.sentences)) * 100

    def rouge(self, manual_summary):
        return rouge_score(self.summary, manual_summary)


if __name__ == '__main__':
    idf = unpickle('idf')
    vocab = unpickle('vocab')

    count = CountVectorizer(stop_words='english', vocabulary=vocab)

    url = 'http://www.upi.com/Odd_News/2016/08/17/Duck-forms-unlikely-friendship-with-depressed-dog-in-Tennessee/1291471449756/?spt=sec&or=on'
    # url = 'https://www.washingtonpost.com/news/morning-mix/wp/2016/08/16/son-of-chicago-cop-home-from-college-to-surprise-his-sick-mom-killed-in-mistaken-identity-shooting/'
    summary_url = 'http://www.newser.com/story/229746/chicago-cops-teen-son-shot-dead.html'

    full_text = get_full_article(url)
    manual_summary = get_summary_and_full_links(summary_url)[0]

    my_sum = Summarizer(vocab, idf, scoring='significance', vectorizer=count)

    my_sum.fit(full_text)
    random_summary_array = my_sum.get_important_sentences(my_sum.random_baseline(my_sum.sentence_vectors))
    random_summary = my_sum.make_summary(random_summary_array)

    print my_sum.summary
    print "Reduction: ", my_sum.reduction, "of original sentences kept"
    print "Rouge: ", my_sum.rouge(manual_summary)
    print "Random Rouge: ", rouge_score(random_summary, manual_summary)
