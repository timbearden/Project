import requests
import re
from bs4 import BeautifulSoup
from summarizer.summarizer import Summarizer
from summarizer.summary_scraping import get_full_article
from summarizer.summarizer_dev import unpickle
from sklearn.feature_extraction.text import CountVectorizer
from outgoing_email import send_email
from newsletter import Newsletter

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
titles = [title.text for title in titles]

gross_links = soup.findAll('link')
links = [re.findall(r'url=(.*)', link.getText())[0].encode('utf-8') for link in gross_links[2:]]

vectorizer = CountVectorizer(stop_words = 'english', encoding = 'utf-8')

summarizer_list = []
for url, title in zip(links, titles):
    try:
        summarizer_list.append(Summarizer(url, vectorizer, title = title))
    except ValueError:
        pass


# idf = unpickle('summarizer/idf')
# vocab = unpickle('summarizer/vocab')
# count = CountVectorizer(vocabulary=vocab, stop_words='english')
#
# summarizer = Summarizer(vocab=vocab, idf=idf, scoring='significance', vectorizer=count)
#
# summaries = []
# reductions = []
# for link in links:
#     article = get_full_article(link[0])
#     summarizer.fit(article)
#     summaries.append(summarizer.summary)
#     reductions.append(summarizer.reduction)


for summarizer in summarizer_list:
    summarizer.summarize()

n = Newsletter(summarizer_list)
n.construct_html()

send_email(n.html)
