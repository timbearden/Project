## Galvanize DSI Final Project Proposal
###### Tim Bearden
July 25, 2016

### Motivation
Staying informed is hard to do when you're very busy. People have to make decisions abut where to spend their time and energy, and in our modern society, more information is being thrown at us than we can ever hope to deal with. These time issues come in two forms: first, what information out there do I actually need to know, and what is the best way for me to get that information quickly and efficiently?

I intend to address both of these issues with a mailing list to automatically send summaries of the ten most popular articles of the day. To do this, I plan on using Twitter popularity to find out what articles are considered important and what are not, and I will use natural language processing to summarize the text of each of the most important articles so that they can be quickly and easily digested.


### Project Idea
1. Scan Twitter's API each day to find links to news articles.
2. Rank these news links based on a formula depending on the number of times tweeted, retweeted and/or liked, and keep the ten most important/popular articles for that day.
3. Develop an automated summarization scheme based on NLP techniques to find the most important sentences based on word importance.
4. Use this process to summarize each of the ten articles.
5. Set up an automatic mailing list to send daily emails with the Title, Summary, and URL of each of the top articles of the day.


### Tools
- Tweepy: Python package to make Twitter API calls.
- MongoDB: To store and manage all of the Twitter data.
- BeautifulSoup: To easily find article text from web links.
- NLTK: For processing article text to rank sentences based on word importance.
- Mailchimp: Automated mailing list service, used to set up the daily emails.  


### Potential issues
- Timing: Do I need to be streaming tweets all day every day to get the information I need, or can I focus on high traffic times?
  - I plan on doing some exploratory data analysis before fully building out the functionality of the project. I will probably stream a day or two's worth of tweets with timestamps to see whether there is a period of time each day that I can use that would be the most efficient.
- How do I deal with fake/promotional accounts, or trolling tweets?
  - Through my early data exploration I hope to find out some indicators of fake accounts. For example, there might be certain locations that fake accounts are registered in, or if a Twitter user has under a certain amount of followers I might exclude them, etc. I will probably also exclude any tweets from corporate accounts.
  - I also plan on fine-tuning my popularity formula to find out what provides the best information on article importance (e.g. maybe promotional accounts might pay for likes and so retweets would be a better indicator of actual popularity, etc.)
- How will I deal with links to non-articles?
  - This issue will be a tough one to deal with. I might be able to use some of the text in each tweet to figure out whether an article is being tweeted out or some other type of URL.
