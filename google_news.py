import requests
import re
from bs4 import BeautifulSoup
from summarizer.summarizer import Summarizer
from summarizer.summary_scraping import get_full_article
from summarizer.summarizer_dev import unpickle
from sklearn.feature_extraction.text import CountVectorizer

# q = raw_input('What would you like to read about? (Put "10" if you just want the \
# top ten news articles) ')
# if q == '10':
#     url = 'https://news.google.com/news?output=rss'
# else:
#     url = 'https://news.google.com/news?q={}&output=rss'.format(q)

url = 'https://news.google.com/news?output=rss'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

titles = soup.findAll('title')[2:]
titles = [title.text.replace("&apos;","'") for title in titles]

gross_links = soup.findAll('link')
links = [re.findall(r'url=(.*)', link.getText()) for link in gross_links]
links = links[2:]

idf = unpickle('summarizer/idf')
vocab = unpickle('summarizer/vocab')
count = CountVectorizer(vocabulary=vocab, stop_words='english')

summarizer = Summarizer(vocab=vocab, idf=idf, scoring='significance', vectorizer=count)

summaries = []
reductions = []
for link in links:
    article = get_full_article(link[0])
    summarizer.fit(article)
    summaries.append(summarizer.summary)
    reductions.append(summarizer.reduction)

for title, summary, reduction, link in zip(titles, summaries, reductions, links):
    print title.upper()
    print '-----------------------------------------'
    print summary
    print '-----------------------------------------'
    print 'Size reduction:', str(round(reduction*100,2)) + '% of original sentences kept'
    print 'URL: ', link[0]
    print '-----------------------------------------'
    print '-----------------------------------------'
