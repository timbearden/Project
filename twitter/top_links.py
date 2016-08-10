import pandas as pd
import numpy as np
from collections import Counter

tweets = pd.read_csv('../data/tweets.csv', encoding='utf8', engine='python')
link_lists = tweets.link

link_splits = map(lambda links: links.split() if links else '', link_lists)
links = [link for link_list in link_splits for link in link_list]
links = [link.replace('[','').replace(']','').replace(',','') for link in links]

link_counts = Counter(links)
