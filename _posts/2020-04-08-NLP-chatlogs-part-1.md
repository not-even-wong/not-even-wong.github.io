---
layout: post
title: "Natural Language Processing of Chatlogs Part 1: Exploratory analysis and dealing with short texts"
date: 2020-04-08
thumbnail: https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/20200408_thumb.jpg
---

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/20200408_thumb.jpg" style="width: 300px" class="border"></p>

I haven't made any new posts for several months. That's in part because things were getting busy. Other than my regular workload, I had the opportunity to work with a company that runs an online university course on <a href="https://en.unesco.org/themes/futures-literacy" target="_blank">Futures Literacy</a>. I spent the past few months developing, refining, and implementing online interaction methods and content delivery, and designing assessment and evaluation tools for the course. Now that the course is coming to an end, I'm shifting more towards data analysis using student work during the course.

I want to write about what I'm doing, but it's tricky because of privacy and data protection. Most likely it'll be ok to discuss my approaches and their limitations, and show some of the aggregated results, but I can't show any specific posts, and especially can't reveal any personal information. I'm worried about displaying code also, because it might show some specific info, but I might consider uploading the code with things like stopwords blanked out.

With this disclaimers aside, I'll jump in and talk about the work I'm doing. There's three key components of student work that I'm analysing:

<ol>
<li>Weekly online live discussion sessions, using Slack (essentially a private chatroom)</li>
<li>Weekly blogposts, using Edublogs (which runs on Wordpress)</li>
<li>Feedback surveys</li>
</ol>

Each of those is a pretty big endeavour on its own. I've only done some preliminary work on the blogposts, and the survey results are not in yet, but I'll be talking about what I've been doing with the discussion sessions. As part of the course requirements, all students are required to sign up for a 1-hour slot each week to discuss that week's content. These sessions are driven by a facilitator based on a rough script, and centered around 2-3 key questions or themes. 

My objective was to find some way to characterise the discussion: what are the students talking about? Once that's done, I can then automate extraction of information. For example, how do the discussion themes shift over the course of each session or across various weeks? 

<b><u>Characterising the discussion</u></b>

Using this chatlog data posed a significant challenge. Each message is usually roughly 1-2 sentences long:

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/1%20post%20length.png" class="border"><br><div class="fineprint">Each bin has a width of approximately 10</div></p>

Slightly less than half of all messages have less than 20 words, and approximately 5/6 of all messages have less than 40 words. Also, in general the longer posts are from facilitators (and copied from the script), and I'm more interested in what the participants are saying. In fact, I'm guilty of making the longest few posts... being verbose is one of my primary flaws.

Anyway, this means that standard LDA topic modeling would not apply. LDA (latent dirichlet allocation) assumes that each text may have a collection of topics, and tries to allocate words to topics and topics to texts accordingly. However, with each text being so short, most of them don't actually have a topic (e.g. "I agree!"), and skimming through the rest, most of the messages indeed only discuss one topic at a time.

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

Firstly, this approach would give me hard clusters. However, while LDA failed since the short texts violate the assumption that each text contains multiple topics, this does not mean that <b><i>all</i></b> texts only have one topic. Furthermore, some texts would have no relevant topic: there are a lot of messages saying things like "I agree!". I didn't think it was appropriate for that to skew my results. In fact, when I did a trial run anyway, I did obtain a cluster containing only variants of "Good morning". Admittedly, I'm probably also slightly biased, since in my experience, clustering word vectors often doesn't seem to give me good results.

In addition, I wasn't sure what kind of word vectors would be appropriate. Since each text is really short, a small absolute variation in term frequency is a large percentage variation. In the end, I went with tf-idf, but using a sublinear tf (i.e. each additional count has a decreasing contribution to the term frequency score). You can see what that looks like here: <a href="https://www.wolframalpha.com/input/?i=1%2Blog%28x%29+from+0+to+100" target="_blank">https://www.wolframalpha.com/input/?i=1%2Blog%28x%29+from+0+to+100</a>

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20200408/3%20sublinear%20tf.png" class="border"></p>


I did try it out, anyway, and while clustering the word vectors did give me some recognisable topics, I felt that it was not precise enough.

The other thing I tried with word vectors was to attempt to find cosine similarity between the discussions and the online lesson content. Each week, students are supposed to read the lesson content and then attend a 1-hour discussion session based on that content. The sessions are facilitated based on a script to discuss key questions relating to that week's content. Therefore, in theory, the discussion should closely match the online content.

However, I realised that the online content was mostly delivered through videos, with some introductory texts linking the videos together. I tried this with the introductory text anyway, and found no meaningful relationship between the discussions and introductory texts, which was to be expected.

<b><u>What I did in the end</u></b>

Eventually, while discussing this project with a friend who worked on analysing Reddit posts, that friend recommended concatenating messages made by the same user and performing topic modeling on those combined texts. By doing so, each text is long enough to use traditional LDA-based topic modeling. Apparently this worked very well for the analysis of Reddit posts.

Using this method and performing LDA topic modeling with gensim, I did manage to capture quite meaningful topics from the discussion. Then, using the topics <i>obtained by analysing concatenated messages</i>, I applied the model to each individual message to get the topic weights for each message, hence bypassing the problem of not being able to obtain meaningful topics from short texts while still being able to characterise these short texts.

In the end, I settled on 8 topics:

<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Topic</th>
<th title="Field #2">Label</th>
<th title="Field #3">Top words</th>
</tr></thead>
<tbody><tr>
<td align="right">1</td>
<td>Automation of work</td>
<td>AI, work, increas(e), job, think, robot, drug, worker, human, replac(e), incom(e)</td>
</tr>
<tr>
<td align="right">2</td>
<td>Futures literacy</td>
<td>Technolog(y), futur(e), understand, human, societ(y), think, chang(e), world, develop, live, AI</td>
</tr>
<tr>
<td align="right">3</td>
<td>Access to online learning</td>
<td>Onlin(e), learn, univers(e), think, student, access, peopl(e), educ(ation), lectur(e), degre(e), teach</td>
</tr>
<tr>
<td align="right">4</td>
<td>Meaningful work</td>
<td>Work, daunt(ing), home, person, UBI &lt;i&gt;(universal basic income)&lt;/i&gt;, meet, peopl(e), contribut(e), paid, excit(e), compan(y)</td>
</tr>
<tr>
<td align="right">5</td>
<td>Democracy</td>
<td>Think, peopl(e), media, democrac(y), social, use, polit(ics), view, data, good, govern</td>
</tr>
<tr>
<td align="right">6</td>
<td>Automation of work</td>
<td>Job, think, work, human, AI, futur(e), machin(e), excit(e), replac(e), develop, labour</td>
</tr>
<tr>
<td align="right">7</td>
<td>Facilitator scripts</td>
<td>Discuss, tell, today, futur(e), week, convers(ation), start, pleas(e), forum, welcom(e), question</td>
</tr>
<tr>
<td align="right">8</td>
<td>Forced online learning during COVID-19 shutdown</td>
<td>Learn, educ(ation), epidem(ic), everyon(e), MOOC, assign, week, familiar, commerci(al), opinion, cover</td>
</tr>
</tbody></table>

Do note that the topic labeling was informed by my experience working on the content: the scriptwriting process, as well as observing or facilitating each session, allowed me to recognise exactly which aspects of the discussion these topics come from. To verify these topics, I viewed the top weighted messages for each topic, and found that they indeed were discussing that topic.

Topics 1 and 6 have a heavy overlap (which can be seen in the pyLDAvis visualisation). However, when I tried reducing the number of topics and changing the random seed, other topics vanished instead of having 1 and 6 merge. While the other topics have some overlap, I feel that they are distinct enough that they should be preserved. Therefore, I left it as 8 topics with 1 and 6 overlapping.

I also noted that these were very general topics. It is especially evident when viewing the top posts for each topic. For example, viewing the top posts about automation of work shows posts discussing it in various contexts or with different opinions, and I do want to be able to obtain data on that. However, increasing the number of topics in gensim simply returned more duplicates of these 7 key topics, so I'd have to use a different method - this will be discussed in a future post!

So far, I've tried using various methods to deal with the short text problem, and in the end, it looks like traditional LDA still worked the best - albeit with a slight variation to the input. Now that I'm able to characterise each short text based on what broad themes it's talking about, I can use that to describe the conversation. However, I'm not done with that analysis - so that'll come in a part 2!
