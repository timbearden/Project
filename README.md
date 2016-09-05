# Automated News Summarizer
## Galvanize DSI Cohort 15 Final Project
### Tim Bearden

## Quick find
1. [Motivation](#motivation)
2. [Summary Process](#summary-process)
  - [Evaluation](#evaluation)
  - [Selecting a length](#selecting-a-length)
  - [Scoring Process](#scoring-process)
  - [Getting the top articles](#getting-the-top-articles)
4. [Where to go from here](#where-to-go-from-here)
3. [Example Entry](#example-entry)

## Motivation
It can be really hard to stay informed. With the amount of information that's thrown at us on a daily basis, unless you just want to skip out on the endeavor, you have to spend some time trying to wade through it all. I wanted to create something to automate this whole process away.

My goal was to create an automated e-mail newsletter that summarized the most important articles of the day. It would scour the internet to find what articles were generating the most attention, then extract the most important from each of those articles into an easy to read summary. Then it would put that summary into a convenient daily email, along with the article link and some indication of how much longer the article is, in case I wanted to know more.

## Summary Process
The first step was to build the summarizer. The process I would use to summarize an article would be as follows:

1. Find out what words were most important in the article based off some [scoring metric](#scoring-process).
2. Split the article into its component sentences.
3. Score each sentence based off how many of those important words were in that sentence.
4. Select the [n](#selecting-length) highest scoring sentences.
5. String those sentences in order into a coherent summary.

#### Evaluation
I tested out many different scoring processes along the way to fine-tune my summarizer. In order to actually evaluate which ones where better, I had to have some way to evaluate the quality of a summary. The most popular evaluation metric for automated summarizers is called ROUGE-N. ROUGE-N compares automated summaries to human-written summaries, by finding the fraction of n-grams that appear in a human-written summary that also appear in the automated summary; it is essentially a recall score for summaries.

To evaluate each of my models, I scraped around 1800 news summaries and their corresponding articles from [newser.com](#https://newser.com), a website that writes summaries of news articles. I compared each of my models to a baseline model that randomly selected sentences from the article, using a ROUGE-2 score (with bi-grams).

#### Selecting a length
One issue that came up was deciding how much to reduce each summary by. Originally I had planned on picking either a certain number of sentences, or a certain fraction of them, but I realized that there might be better, more quantitative ways I could go about it.

I ended up calculating a 'fractional importance' of the summary, and setting a threshold to stop adding sentences once I passed that threshold. Whatever scoring metric I would use, I would calculate for the entire document, and then calculate what fraction of that total score each additional top-scoring sentence added to the total score. Through experimentation, I found that setting a threshold of 50% of the articles overall score provided the best trade-off between quality (ROUGE) and effectiveness (how much it reduced the article). With this threshold, most of my summaries ended up reducing to about 20-30% of their original length, with flattening increases in ROUGE.

#### Scoring Process
I tested out several different scoring processes for my model. One of the simplest ones turned out to be the best; I ended up going with a summarizer that calculated the term-frequencies for all of the words in the article, minus stop words, and then summed up those term counts for each time it showed up in that specific sentence. For example, if the word "cat" showed up three times in the article, and "dog" showed up twice, and there was the sentence "The cat chased the dog" appeared, that sentence would have a score of 3 + 2 = 5. I was very surprised at how well this worked compared to all of my other methods, as this seemed to be a relatively crude approach, but in the end it had the best performance both in terms of quality, and effectiveness.

I also experimented with a similar tactic, but using tf-idf vectors instead of straight article counts. Tf-idf takes the vector of term-frequencies similar to the one I used in my final model, then multiplies it by the inverse document frequency, which is a way of down-weighting words that show up frequently in many different articles. In one version, I used a background corpus of around 10,000 New York Times articles to calculate my inverse document frequency. I had thought this would be my best method, but while it did a great job of making really short summaries, it sacrificed the quality of the summary to do so. I also tried calculating tf-idf where I treated each sentence as a document, and the overall article as the corpus of documents. This created surprisingly good summaries, but they were much longer than in any of my other methods.

I also tested out different aggregation methods than summing. I tried calculating mean word scores for each sentence using these same vectorization techniques. This would capture more short sentences, which helps keep the summary quick, but wouldn't create very great quality summaries. I also tested out using the cosine similarity of the tf and tf-idf vectors of each sentence to the vector for the entire article, though this was not possible for the single-document version of the tf-idf score. This turned out not to be very effective either.

#### Getting the top articles
Originally, I had planned on using Twitter to find the most important articles of the day. My hypothesis was that the biggest news articles are going to get the highest of some combination of tweets, retweets, and likes. As it turns out, if you use Twitter to measure the popularity of different links, the only results that you will get will be articles about K-pop.

As a quick fix, I am currently using Google News' RSS feed. For further expansion of this project, I would like to develop a better, more personalized way to find the most important articles of the day.  

## Where to go from here
- Develop a better way to find the top news articles.
- Create a way to track specific subjects, to make personalized e-mail newsletters.

## Example Entry
#### First Read: Clinton Has Owned the Airwaves in the General Election
Exactly two months ago, Hillary Clinton's campaign went up with its first general election TV ads in battleground states, and in that time it has spent $61 million over the airwaves, while pro-Clinton outside groups have chipped in an additional $43 million. What is Trump doing with his campaign money (after the New York Times reported two weeks ago that Trump and the GOP had raised a combined $82 million last month)? Oh, and get this: The Green Party's Jill Stein ($189,000) and Libertarian nominee Gary Johnson ($15,000) have spent more on ads than the Trump campaign ($0) in this general election.

Trump outside groups: $12.4 million
Total Team Trump: $12.4 million

Clinton 50%, Trump 41%: Meanwhile, the latest weekly national NBC.SurveyMonkey online tracking poll shows Hillary Clinton leading Donald Trump by nine points, 50%-41% -- virtually unchanged from last week's 51%-41% advantage for Clinton. Just 17 percent of all voters say that Trump has the personality and temperament to serve effectively as president. Even among Republican and Republican-leaners, only 19 percent said Trump has the personality to serve effectively. "[T]he national security framework he described was so contradictory and filled with so many obvious falsehoods that it's virtually impossible to tell what he would do as president… That's because Trump previously supported every single foreign policy decision he now decries. Despite claiming daily that he opposed the Iraq War from the start, Trump endorsed deposing Saddam Hussein in a 2002 interview and there's no record of him opposing the war until after it had began. Hillary Clinton holds a voter-registration even in Philadelphia at 1:15 pm ET… Tim Kaine hits North Carolina… Donald Trump holds a rally in West Bend, WI at 8:30 pm ET… And Mike Pence is in New Mexico.

**Size Reduction:** 21.95% of original sentences kept

**Url:** http://www.nbcnews.com/politics/first-read/first-read-clinton-has-owned-airwaves-general-election-n631706

Rouge:  0.195804195804
Random Rouge:  0.0559440559441

## Sources
- Das, Dipanjan, and André F.T. Martins. "A Survey on Automatic Text Summarization." N.p., 21 Nov. 2007. Web
- Lin, Chin-Yew, and Eduard Hovy. "The Automated Acquisition of Topic Signatures for Text Summarization." Proceedings of the 18th Conference on Computational Linguistics (2000): n. pag. Print.
- Lin, Chin-Yew. "ROUGE: A Package for Automatic Evaluation of Summaries." (n.d.): n. pag. Web.
- Lin, Chin-Yew. ROUGE. ROUGE: Recall-Oriented Understudy of Gisting Evaluation. N.p., n.d. Web.
- "Newser: Headline News Summaries." Newser: Headline News Summaries. Ed. Neal Colgrass. Newser, n.d. Web. 14 Aug. 2016.
