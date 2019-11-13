---
layout: post
title: "A layman primer to topic modeling"
date: 2019-11-10
thumbnail: https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191110/20191110_thumb.png
---
<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191110/20191110_thumb.png" style="width: 300px" class="border"></p>

After deciding to put this portfolio together, I realised that not everyone knows what topic modeling is.

Well, now that datamining is all the rage, you see lots of jargon getting thrown around. You might have heard the terms "natural language processing" or "text analysis" being used, or "machine learning", "AI", and so on. These are subtly different!

If I were to summarise them, I would say...

Natural Language Processing is a branch of data science (i.e. the science behind using huge amounts of data) that focuses on trying to understand and interpret text like a human would. Sometimes, this may use machine learning, which is essentially just very advanced applied statistics. Machine learning, in a very simplified nutshell, just calculates the probabilities of certain outcomes based on data, and uses this to predict outcomes when given new data - it has 'learnt' from the training dataset, and tries to generalise what it has 'learnt' to new information... which is not very different from what we humans do!

Machine learning can broadly be classified into supervised or unsupervised learning. That just means whether or not the person writing the algorithm provides information called "labels" for the data. An unsupervised algorithm will look at the raw data without any 'understanding' of what it means. It'll just see a bunch of numbers, and try to process them without context. A supervised algorithm will be told what the 'correct' results are: for example, a collection of images might be given the labels "car", "traffic light", "storefront" (and I believe this is why Captchas are asking you for this information nowadays!). The algorithm will then try to process the data to sort out what characteristics of the data can be used to identify it according to the labels, and hopefully it can then be used to analyse new (unlabelled) data.

Topic Modeling is (usually) a kind of unsupervised learning. It's a way of using an algorithm to look at a large collection of text (known as a 'corpus'), and identify what topics might exist. The assumption is that if you have a given number of texts (e.g. articles, essays, etc.), these texts will discuss a number of topics (e.g. science, music, politics), and each of these topics will have words associated with them (e.g. politics may be associated with words like president, senator, voting, party). Without going into the details, topic modeling algorithms identify topics by identifying what words tend to be used together.

Of course, since it's unsupervised, the algorithm won't know what the topic should be called. That's up to you, as the user! For example, running an algorithm on a corpus of newspaper articles, perhaps it might identify a few topics associated with these words:

<ul>
<li>research, discovery, lab</li>
<li>guitar, concert</li>
<li>president, senator, voting, party</li>
</ul>

As a human reader, you might then decide to label these topics as "science", "music" and "politics" respectively.

After that, you could then determine the "weight" of each topic for each article in the corpus! For example, an article about the president of a particular country declaring subsidies for STEM research might have a higher weight of topics 1 and 3, but negligible weight for topic 2. 

Data and analysis like this can be used to tell you, very quickly, what the corpus as a whole is about. However, topic modeling is a tool that's only as good as the user. It's not always easy to decide what parameters to use to run the analysis, or how to interpret the output. Furthermore, your analysis will only be as good as (or worse than) your raw data. So you can expect to spend a lot of time cleaning your input data to ensure that it's free from typos or doesn't contain irrelevant data. Machines can also find it really difficult to differentiate between similar words used in different contexts, or other weird quirks of language that we humans have somehow learnt to deal with.

It might also be useful to know the following terms:


<ul>
<li>Stemming and lemmatization: This is are ways of reducing words to their root form for easy comparison. For example, two articles using the words "running" and "runned" respectively are presumably talking about the same thing.</li>
<li>Clustering: This is a family of unsupervised machine learning techniques that group samples of data together based on various similarity metrics. For example, if two sets of data are (1, 0, 0) and (2, 0, 0), they would be more likely to be grouped with each other instead of with another set of data (0, 0, 5)</li>
<li>Text data vectorization: This is a way of making use of word counts in a document and other tricks to convert text data into numbers that can be more easily processed by machine learning algorithms.</li>
</ul>
