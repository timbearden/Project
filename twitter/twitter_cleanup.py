import pandas as pd
import json

tweets_data_path = 'data/twitter_data.txt'

tweets_data = []
with open(tweets_data_path, "r") as f:
    for line in f:
        try:
            tweet = json.loads(line.strip())
            if 'text' in tweet:
                tweets_data.append(tweet)
        except:
            continue

tweets = pd.DataFrame()

tweets['text'] = map(lambda tweet: tweet.get('text',''), tweets_data)
tweets['time'] = map(lambda tweet: tweet.get('created_at',None), tweets_data)
tweets['lang'] = map(lambda tweet: tweet.get('lang',None), tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] else None, tweets_data)
tweets['retweets'] = map(lambda tweet: tweet.get('retweet_count',0), tweets_data)
tweets['favorites'] = map(lambda tweet: tweet.get('favorite_count',0), tweets_data)

split_text = [text.split() for text in tweets['text']]
url_list = []
for text in split_text:
    url_list.append([word for word in text if word[:4] == 'http'])

tweets['link'] = url_list

tweets = tweets[tweets['lang'] == 'en']

tweets.to_csv('data/tweets.csv', encoding='utf-8')
