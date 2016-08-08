import requests
import threading
from bs4 import BeautifulSoup
from newspaper import Article
from pymongo import MongoClient
from time import sleep


def soup_it_up(url):
    sleep(0.25)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def get_summary_links(url):
    soup = soup_it_up(url)
    link_tags = soup.find_all('a', class_='b')
    summary_urls = []
    for tag in link_tags:
        summary_urls.append('http://www.newser.com' + tag.attrs['href'])
    return summary_urls


def get_summary_and_full_links(url):
    soup = soup_it_up(url)
    story = soup.find_all('p', class_='storyParagraph')
    url_tags = soup.select('p a')
    text = ''
    for paragraph in story:
        text += paragraph.text
    text = text.replace("\r", " ")
    article_urls = []
    for tag in url_tags:
        article_urls.append(tag['href'])
    return text, article_urls


def get_full_article(url):
    a = Article(url, language='en')
    a.download()
    a.parse()
    return a.text


def get_all_and_store(url):
    summary, article_urls = get_summary_and_full_links(url)
    article_urls = [link for link in article_urls if 'newser' not in link and 'youtube' not in link and 'twitter' not in link and 'soundcloud' not in link]
    if article_urls:
        full_article = [get_full_article(link) for link in article_urls]
    else:
        full_article = " "
    coll.insert_one({'summary_url':url, 'summary':summary, 'full_url':article_urls, 'full_text':full_article})


if __name__ == '__main__':
    base_url = "http://www.newser.com/siteindex/story/2015/"

    index_url_list = []
    for month in xrange(4,7):
        for page in xrange(1,21):
            url = base_url + str(month) + "/" + str(page) + ".html"
            index_url_list.append(url)


    summary_urls = []
    for url in index_url_list:
        summary_urls.extend(get_summary_links(url))
    summary_urls[1300:]

    mongo_client = MongoClient()
    db = mongo_client.g_project_data
    coll = db.test_data

    for url in summary_urls:
        t = threading.Thread(target=get_all_and_store, args=(url,))
        t.start()

    mongo_client.close()
