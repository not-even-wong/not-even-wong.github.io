---
layout: post
title: "Natural Language Processing of Chatlogs Part 1: Exploratory analysis and dealing with short texts"
date: 2020-04-08
thumbnail: https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/20200408_thumb.jpg
---

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/20200408_thumb.jpg" style="width: 300px" class="border"></p>

<div class="fineprint">Images and tables are generally scrollable if not displayed in full.</div>

I haven't made any new posts for several months. That's in part because things were getting busy. Other than my regular workload, I had the opportunity to work with a company that runs an online university course on <a href="https://en.unesco.org/themes/futures-literacy" target="_blank">Futures Literacy</a>. I spent the past few months developing, refining, and implementing online interaction methods and content delivery, and designing assessment and evaluation tools for the course. Now that the course is coming to an end, I'm shifting more towards data analysis using student work during the course.

I want to write about what I'm doing, but it's tricky because of privacy and data protection. Most likely it'll be ok to discuss my approaches and their limitations, and show some of the aggregated results, but I can't show any specific posts, and especially can't reveal any personal information. I'm worried about displaying code also, because it might show some specific info, but I might consider uploading the code with things like stopwords blanked out.

With this disclaimers aside, I'll jump in and talk about the work I'm doing. There's three key components of student work that I'm analysing:

<ol>
<li>Weekly online live discussion sessions, using Slack</li>
<li>Weekly blogposts, using Edublogs (which runs on Wordpress)</li>
<li>Feedback surveys</li>
</ol>

Each of those is a pretty big endeavour on its own. I've only done some preliminary work on the blogposts, and the survey results are not in yet, but I'll be talking about what I've been doing with the discussion sessions. As part of the course requirements, all students are required to sign up for a 1-hour slot each week to discuss that week's content. These sessions are driven by a facilitator based on a rough script, and centered around 2-3 key questions or themes. 

My objective was to find some way to characterise the discussion: what are the students talking about? Once that's done, I can then automate extraction of information. For example, how do the discussion themes shift over the course of each session or across various weeks? 

<b><u>Characterising the discussion</u></b>

However, using chatlog data poses a challenge. Each text is usually roughly 1-2 sentences long:

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/1%20post%20length.png" class="border"><br><div class="fineprint">Each bin has a width of approximately 10</div></p>

Slightly less than half of all messages have less than 20 words, and approximately 5/6 of all messages have less than 40 words. Also, in general the longer posts are from facilitators (and copied from the script), and I'm more interested in what the participants are saying. In fact, I'm guilty of making the longest few posts... being verbose is one of my primary flaws.

Anyway, this means that standard LDA topic modeling would not apply. LDA (latent dirichlet allocation) assumes that each text may have a collection of topics, and tries to allocate words to topics and topics to texts accordingly. However, with each text being so short, most of them don't actually have a topic (e.g. "I agree!"), and skimming through the rest, they indeed generally only discuss one topic at a time.

<b><u>Reviewing the literature on chatlog analysis</u></b>

I spent quite a while trying to identify methods to get around this. My first thought was to look for NLP analyses of chatlogs in the literature. Interestingly, these seemed to mostly revolve around the following topics:

<ul>
<li>Customer service chatlogs - however, these datasets are MUCH larger than what I have.</li>
<li>Use of chatrooms in education - I'm using chatrooms in education, but they aren't doing the kind of analysis I'm looking for.</li>
<li>Bot AI research - definitely not relevant.</li>
<li>Machine learning to identify criminals - too much of a black box! I want to datamine the messages, rather than use it for prediction.</li>
</ul>

The literature on this didn't really provide me with the information I needed - how do I datamine a collection of short texts for information that I can use to characterise each text? 

<b><u>Visualising the overall discussion</u></b>

Around this point, I also tried to characterise each overall session. At this time, I was also taking an additional course on information visualisation, and came across an interesting approach known as "Sententrees". You can see the authors' writeup here: <a href="https://github.com/twitter/SentenTree" target="_blank">https://github.com/twitter/SentenTree</a>

Their work essentially expands on the idea of a wordcloud. Other than using size to visualise word frequency, it also tries to visualise  n-grams frequency. The prime example they gave on their documentation, which was sourced using a very large number of tweets, looked quite impressive:

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/SentenTree.png" style="width: 600px" class="border"></p>

Unfortunately, when I used the algorithm on my much smaller dataset, the results leave much to be desired:

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/2%20sententree.png" style="width: 600px" class="border"></p>

I realised that this was probably due to two reasons: firstly, I didn't have enough samples, and furthermore, due to the nature of the discussion, I'd be very unlikely to get a lot of repeating n-grams. This is because successive messages may be about the same general topic, but have very different content: for example, a discussion about AI would have different people bringing in different opinions about AI and how it applies to various aspects of their lives.

<b><u>Short Text Topic Modeling</u></b>

After that, I looked for Twitter analysis and Short Text Topic Modeling (STTM). Generally, STTM approaches assume that each text only contains one topic. However, different authors may use different algorithms to assign words to topics. To my dismay, I found that although STTM has been in development over the past decade, there still isn't really a definite tried-and-tested approach that works (unlike the use of LDA for longer texts). Lots of people have tried various approaches, to different levels of success. Unfortunately, I don't feel confident enough in my expertise and the literature to implement these methods.

<b><u>Word vector analysis</u></b>

Another approach that might have been useful would be to try to characterise the texts using word vectors, and then cluster them, using the clusters to characterise the discussion. By looking at the top words for each cluster or skimming the posts, I could label these clusters. However, I had two major concerns.

Firstly, this approach would give me hard clusters. However, while LDA failed since the short texts violate the assumption that each text contains multiple topics, this does not mean that <b><i>all</i></b> texts only have one topic. Furthermore, some texts would have no topic: there are a lot of messages saying things like "I agree!". I didn't think it was appropriate for that to skew my results. In fact, when I did a trial run anyway, I did obtain a cluster containing only variants of "Good morning!"

In addition, I wasn't sure what kind of word vectors would be appropriate. Since each text is really short, a small absolute variation in term frequency is a large percentage variation. In the end, I went with tf-idf, but using a sublinear tf (i.e. each additional count has a decreasing contribution to the term frequency score).

I did try it out, anyway, and while clustering the word vectors did give me some recognisable topics, I felt that it was not precise enough.

The other thing I tried with word vectors was to attempt to find cosine similarity between the discussions and the online lesson content. Each week, students are supposed to read the lesson content and then attend a 1-hour discussion session based on that content. The sessions are facilitated based on a script to discuss key questions relating to that week's content. Therefore, in theory, the discussion should closely match the online content.

However, I realised that the online content was mostly delivered through videos, with some introductory texts linking the videos together. I tried this with the introductory text anyway, and found no meaningful relationship between the discussions and introductory texts, which was to be expected.



