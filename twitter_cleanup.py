import pandas as pd
import json

tweets_data_path = 'twitter_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

tweets = pd.DataFrame()

tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['time'] = map(lambda tweet: tweet['created_at'], tweets_data)
tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)
tweets['retweets'] = map(lambda tweet: tweet['retweet_count'], tweets_data)
tweets['favorites'] = map(lambda tweet: tweet['favorite_count'], tweets_data)

split_text = [text.split() for text in tweets['text']]
url_list = []
for text in split_text:
    url_list.append([word for word in text if word[:4] == 'http'])

tweets['link'] = url_list

tweets = tweets[tweets['lang'] == 'en']

tweets.to_csv()
