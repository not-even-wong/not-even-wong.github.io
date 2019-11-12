---
layout: post
title: "Analysing similarities between recipes with topic modeling"
date: 2019-11-12
---

I'm a Singaporean, which means that appreciating the vast diversity of food is a way of life for me. Growing up in a melting pot of cultures and travelling overseas to savour the taste of various local delicacies does that for you.

Sometimes you hear people say things like "This doesn't taste authentic!" 

Well, then what makes something authentic? As an avid amateur chef, my best guess is that (other than sociocultural factors such as who's cooking it and where you're eating it) it's in the ingredients and cooking methods. For example, I don't expect to see Chinese recipes call for letting dough rise before baking, and neither do I expect to see thinly sliced raw fish in American recipes. But I'd like to quantify this.

Ideally, I'd love to be able to interview countless chefs and diners to collate their thoughts about different cultural experiences. Ideally, I'd also love to have unlimited time and funding and research assistants. But we can't always get what we want.

So I started my project by searching for a database of recipes that are conveniently labelled by origin or type. Eventually, I found this website:

<a>https://www.recipesource.com/</a>

Well, that's perfect! Every recipe here is already in text form, and categorised either by country of origin or type of food! 

<i>Note: we've to be careful about the validity of the labels here. This website is run by a team of volunteers and recipes can be submitted to the database; these labels may be influenced by their cultural backgrounds and awarenesses. Still, it's an impressive collection and the recipes, as far as I can tell, are reasonably accurately cataloged</i>

I wrote a quick script in Python to generate a list of all links to recipes or categories from the main page, and iteratively go to each page to either download the recipe there, or generate a new list of links on that subpage. By the end of it, I had saved a csv file with one entry for each page containing the URL of that page, and the text of the page.

The reason why I chose to save the URL is twofold: firstly, it lets me know where the text originally came from if I need to check details, and secondly, this webpage has exceptionally good folder management: the URL itself encodes the label describing what categories and subcategories of country or type of food this recipe falls under! Therefore, by running another short script, I obtained the appropriate label for each and every recipe.

This yielded something in the range of 54 thousand recipes in around 180 subcategories.

<i>Later analysis demonstrated that this data needed a large amount of cleaning. Over the course of this project, I removed a lot of irrelevant terms, as well as standardised some wording. For example, alternate methods of spelling, use of "chili" vs "chilli", replacing abbreviations such as "tsp" to "teaspoon", and so on. It took many iterations of cleaning before I started to get decent results.</i>

Well, here's where the fun begins. I started off by performing k-means clustering on the text of each individual recipe. Quickly scanning through these recipes suggested that the clusters are reasonably accurate. I then clustered categories of recipes based on the number of constitutent individual recipes in each cluster. This also gave me reasonable results, but the approach seemed questionable.

After that, I decided to aggregate all recipes in each category into a single document. That reduced the number of documents from ~54000 to ~180. I then used heirarchical clustering on the tf-idf scores of each document, with the aim of determining which categories of recipes were closest to each other.

This was the result of my clustering:

<div style="height: 400px; width: auto; border: 1px solid #ccc; overflow:scroll;">
<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/heirarchy%20of%20recipes%20tfidf.png">
</div>

