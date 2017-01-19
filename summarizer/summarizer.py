from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk import sent_tokenize, word_tokenize
from newspaper import Article
import re


class Summarizer(object):

    def __init__(self, url, vectorizer, title = None):
        self.url = url
        self.article = ''
        self.title = title
        self.sentences = None
        self.summary = ''
        self.vectorizer = vectorizer
        self.reduction = 0.0
        self.rouge_score = 0.0
        self.formatted = ''

    def get_article(self):
        a = Article(self.url)
        a.download()

        try:
            a.parse()
            self.article = self.clean_text(a.text).encode('utf-8')
            if self.title is None:
                self.title = a.title.encode('utf-8')
        except ValueError:
            self.article = 'Parsing Error'
            self.title = 'Parsing Error'

        self.sentences = np.array(sent_tokenize(self.article))


    def summarize(self):
        self.get_article()

        full_vec = np.array(self.vectorizer.fit_transform([self.article]).todense())[0]
        vocab = self.vectorizer.fit([self.article]).vocabulary_

        score_arr = []
        for sentence in self.sentences:
            words = word_tokenize(sentence)
            score = np.sum([full_vec[vocab[word.lower()]] for word in words if word.lower() in vocab.keys()])
            score_arr.append(score)
        score_arr = np.array(score_arr)

        sort_idx = np.argsort(score_arr)[::-1]
        cumulative_importance = np.cumsum(score_arr[sort_idx] / float(np.sum(score_arr)))
        top_n, = np.where(cumulative_importance < 0.5)
        important_sentence_idx = sort_idx[top_n]
        sentence_idx = np.sort(important_sentence_idx)

        summary_array = self.sentences[sentence_idx]
        self.summary = ' '.join(summary_array).replace('|', '.')
        self.reduction = len(summary_array) / float(len(self.sentences))


    def format_summary(self):
        self.formatted += self.title + '\n'
        self.formatted += '-' * 79 + '\n'
        self.formatted += self.summary + '\n'
        self.formatted += '-' * 79 + '\n'
        self.formatted += 'Size reduction: ' + str(round(self.reduction * 100, 2)) + '% of original sentences kept.\n'
        self.formatted += 'URL: ' + self.url.encode('utf-8') + '\n'
        self.formatted += '-' * 79 + '\n'
        self.formatted += '-' * 79 + '\n'


    def clean_text(self, text):
        '''
        INPUT: string
        OUTPUT: string

        Cleans up the text to eliminate sentences that are not actually in the
        article, keeps quotes in tact for when the text is tokenized by sentence.
        '''
        split_text = text.split(u'\u201c')
        for i in xrange(1, len(split_text)):
            more_split = split_text[i].split(u'\u201d')
            more_split[0] = more_split[0].replace('.', '|')
            split_text[i] = u'\u201d'.join(more_split)
        new_text = u'\u201c'.join(split_text)
        new_text = re.sub(r'(Advertisement.*?\n)', '', new_text)
        new_text = re.sub(r'(Photo.*?\n)', '', new_text)
        new_text = re.sub(r'(?<=[A-Z])\.', '', new_text)
        new_text = re.sub(r'(Related.*?\n)', '', new_text)
        new_text = u'\n'.join([sentence for sentence in new_text.split(u'\n') if u'.' in sentence])
        return new_text


if __name__ == '__main__':
    url = 'https://www.washingtonpost.com/world/europe-leaders-shocked-as-trump-slams-nato-eu-raising-fears-of-transatlantic-split/2017/01/16/82047072-dbe6-11e6-b2cf-b67fe3285cbc_story.html'
    a = Article(url)
    a.download()
    a.parse()

    vectorizer = CountVectorizer(stop_words = 'english', encoding = 'utf-8')

    summarizer = Summarizer(url, vectorizer)
    summarizer.summarize()
    summarizer.format_summary()
