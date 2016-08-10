import requests
import re
from bs4 import BeautifulSoup

q = ''
url = 'https://news.google.com/news?{}output=rss'.format(q)

r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

titles = soup.findAll('title')[2:]
titles = [title.text for title in titles]

gross_links = soup.findAll('link')
links = [re.findall(r'url=(.*)', link.getText()) for link in gross_links]
links = links[2:]
