---
layout: post
title: "Analysing similarities between recipes with topic modeling"
date: 2019-11-12
---
<div class="fineprint">The Python code used for this post can be found here: <a>https://github.com/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191112/recipe-analysis.py</a></div>

<div class="fineprint">Images and tables are generally scrollable if not displayed in full.</div>

I'm a Singaporean, which means that appreciating the vast diversity of food is a way of life for me. Growing up in a melting pot of cultures and travelling overseas to savour the taste of various local delicacies does that for you.

Sometimes you hear people say things like "This doesn't taste authentic!" 

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/authenticity.png" width="50%">

Ok, admittedly I don't really care so much about authenticity. To me, as long as it tastes good, I'm open to giving it a try... sometimes to disastrous results, according to people I cook for.

But when producing creative work, you need to be aware of the cultural literacies of your audience, and cooking is no different. People engage with meals like they do any other experience: bringing a whole lot of baggage and associations. Their experience of the meal is going to be influenced by the lens of their shared experiences. Cooking is a conversation between the chef and the diner through, and across, the language of culture.

So then what makes something authentic? What's the culinary equivalent of a national accent and phenotype?

As an avid amateur chef , my best guess is that (other than sociocultural factors such as who's cooking it and where you're eating it) it's in the ingredients and cooking methods. For example, I don't expect to see Chinese recipes call for letting dough rise before baking, and neither do I expect to see thinly sliced raw fish in American recipes. But I'd like to quantify this.

Ideally, I'd love to be able to interview countless chefs and diners to collate their thoughts about different cultural experiences. Ideally, I'd also love to have unlimited time and funding and research assistants. But we can't always get what we want.

So I started my project by searching for a database of recipes that are conveniently labelled by origin or type. Eventually, I found this website:

<a>https://www.recipesource.com/</a>

Well, that's perfect! Every recipe here is already in text form, and categorised either by country of origin or type of food! 

<div class="fineprint">Note: we've to be careful about the validity of the labels here. This website is run by a team of volunteers and recipes can be submitted to the database; these labels may be influenced by their cultural backgrounds and awarenesses. Still, it's an impressive collection and the recipes, as far as I can tell, are reasonably accurately cataloged</div>

I wrote a quick script in Python to generate a list of all links to recipes or categories from the main page, and iteratively go to each page to either download the recipe there, or generate a new list of links on that subpage. By the end of it, I had saved a csv file with one entry for each page containing the URL of that page, and the text of the page.

The reason why I chose to save the URL is twofold: firstly, it lets me know where the text originally came from if I need to check details, and secondly, this webpage has exceptionally good folder management: the URL itself encodes the label describing what categories and subcategories of country or type of food this recipe falls under! Therefore, by running another short script, I obtained the appropriate label for each and every recipe.

This yielded something in the range of 54 thousand recipes in around 180 subcategories.

<div class="fineprint">Later analysis demonstrated that this data needed a large amount of cleaning. Over the course of this project, I removed a lot of irrelevant terms, as well as standardised some wording. For example, alternate methods of spelling, use of "chili" vs "chilli", replacing abbreviations such as "tsp" to "teaspoon", and so on. It took many iterations of cleaning before I started to get decent results.</div>

Well, here's where the fun begins. I started off by performing k-means clustering on the text of each individual recipe. Quickly scanning through these recipes suggested that the clusters are reasonably accurate. I then clustered categories of recipes based on the number of constituent individual recipes in each cluster. This also gave me reasonable results, but the approach seemed questionable.

After that, I decided to aggregate all recipes in each category into a single document. That reduced the number of documents from ~54000 to ~180. I then used heirarchical clustering on the tf-idf scores of each document, with the aim of determining which categories of recipes were closest to each other.

<div class="fineprint">(tf-idf stands for "term frequency, inverse document frequency". It means that for each word, you count how many times it appears in that particular document, then reduce that amount based on how many documents it appears in. It's a way of reducing the impact of words that appear in too many documents, so your data focuses on words that are unique to a particular document.)</div>

This was the result of my heirarchical clustering:

<div style="height: 400px; width: 510px; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/heirarchy%20of%20recipes%20tfidf.png">
</div>

<div class="fineprint">Note: I used sklearn TfidfVectorizer for the text data vectorization, and scipy for clustering. My maximum and minimum document frequency thresholds were 0.9 and 0.001 respectively.</div>

Well, those clusters seem to make sense. However, I have no clue as to exactly <i>why</i> those categories are clustered together.

So I went on to make use of topic modeling using the Gensim library. <a href="https://not-even-wong.github.io/2019/11/10/layman-intro-topic-modeling.html">I wrote a very quick layman primer to topic modeling here</a>. To some extent, here I am effectively using topic modeling as a means of reducing the dimensionality of the data. I <i>could</i> theoretically compare the tf-idf scores for every single word to analyse the clusters obtained from hierarchical clustering. However, there are just too many words in total. The analysis would make a lot more sense if I reduced the number of possible words to compare (the dimensionality of the data) to a smaller number of topics: that's going from many thousands of dimensions to just the number of topics!

After some experimenting to decide the number of topics, I settled on 35 topics: too few resulted in topics not being specific enough; too many resulted in too much overlap between topics. I used pyLDAvis to visualise these topics:

<a href="https://nbviewer.jupyter.org/github/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191112/pyLDAvis%20visual%20for%20upload.ipynb">View online</a>

<a href="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/lda35.html">Download to desktop</a>

To assign labels, I set λ = 0.6 for a good balance between term frequency and uniqueness, and used the top 15 terms:

<div class="fineprint">Note that the terms obtained from the LDA model have been lemmatized; I've replaced the missing letters in brackets, e.g. juic -> juic(e)</div>

<div style="height: 400px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Topic</th>
<th title="Field #2">Proposed label</th>
<th title="Field #3">Top 15 terms for λ = 0.6</th>
<th title="Field #4">Comments</th>
</tr></thead>
<tbody><tr>
<td>1</td>
<td>Seafood</td>
<td>fish, crab, seafood, fillet, shallot, scallop, lime, thai, mussel, lobster, squid, lemon, juic(e), francisco, san</td>
<td>Presumably, seafood is extremely commonly used in Thai food and associated with the San Francisco fish market. Surprisingly, clam is lower on this list than San Francisco, considering that the clam chowder is famous.</td>
</tr>
<tr>
<td>2</td>
<td>Dry fruits</td>
<td>appl(e), cinnamon, pear, fig, fruit, core, tart, sugar, raisin, peel, fudg(e), prune, nutmeg, dot, apricot</td>
<td>No idea what &quot;dot&quot; refers to. Other words: Apple, cinnamon, pear, fig, fruit, core, tart, sugar, raisin, peel, fudge, prune, nutmeg, dot, apricot, cranberry, granny, brown, slice, lemon, juice, pie, walnut, tablespoon, butter, sprinkle, pastry, water</td>
</tr>
<tr>
<td>3</td>
<td>Fresh fruits</td>
<td>strawberr(y), yogurt, mint, juic(e), lemon, berr(y), orang(e), honey, lime, fruit, fresh, plain, mango, zest, raspberr(y)</td>
<td> </td>
</tr>
<tr>
<td>4</td>
<td>Mexican</td>
<td>chilli, tortilla, mexican, salsa, tomato, cumin, jalapeno, taco, chile, cilantro, jack, green, chop, powder, ounce</td>
<td>Not sure what &quot;jack&quot; refers to here. &quot;Ounce&quot; is commonly used to measure out amounts of sauce. Not sure if &quot;chile&quot; refers to the country, or is a mis-spelling of &quot;chili/chilli&quot;; both seem plausible.</td>
</tr>
<tr>
<td>5</td>
<td>Grilling</td>
<td>eggplant, grill, slice, roast, towel, paper, bacon, broil, place, cut, skewer, platter, broiler, avocado, skin</td>
<td>Eggplant and avocado are outliers. I&#39;m guessing eggplant is here because eggplants are often grilled, and avocado is served with bacon. &quot;Towel&quot; and &quot;paper&quot; probably refers to the paper towels used to let grilled meat &#39;rest&#39; after grilling, paper could also possibly refer to baking paper but this is less likely</td>
</tr>
<tr>
<td>6</td>
<td>Creamy spreads</td>
<td>dip, chees(e), cream, sour, sandwich, tuna, appet(ite), mayonnais(e), salmon, spread, ounce, pita, dill, cracker, hor</td>
<td>&quot;Hor&quot; probably refers to hors d&#39;oeuvr (&quot;d&#39;oeuvr&quot; is 17th on this list; 16th is &quot;blue&quot;, i.e. blue cheese)</td>
</tr>
<tr>
<td>7</td>
<td>Grains</td>
<td>rice, cup, lentil, stir, heat, prepar(e), cook, water, barley, boil, grain, saucepan, method, measur(e), add</td>
<td>&quot;Add&quot; probably is used in the context of &quot;add water after…&quot; or &quot;add [grains] to boiling water&quot;. &quot;measur&quot; is important for cooking grains correctly: sensitive to amount of water used.</td>
</tr>
<tr>
<td>8</td>
<td>Slow cooking</td>
<td>crockpot, crock, pot, oyster, cabbag(e), steak, cooker, slow, clam, hour, beef, cajun, meat, sausag(e), cook</td>
<td>Most of the terms listed here are associated with slow cooking recipes, except &quot;oyster&quot; and &quot;clam&quot;, but these are cooked in pots as well.</td>
</tr>
<tr>
<td>9</td>
<td>Egg &amp; dairy</td>
<td>egg, yolk, butter, custard, milk, popcorn, mixture, white, stir, heat, whisk, melt, cream, beaten, sauc€</td>
<td>Not sure why popcorn is in this list. The only possible reason I can find is that popcorn recipes tend to use butter.</td>
</tr>
<tr>
<td>10</td>
<td>Dough handling</td>
<td>cut, pastr(y), squash, foil, strip, inch, edg(e), knife, place, wrap, end, half, trim, piec(e), aluminum</td>
<td>&quot;Squash&quot; probably refers to the verb, not the noun. &quot;Aluminum&quot; should refer to &quot;aluminum foil&quot;</td>
</tr>
<tr>
<td>11</td>
<td>Soup stock</td>
<td>soup, stock, bay, simmer, carrot, broth, leek, leaf, add, celer(y), onion, stew, pot, boil, bring</td>
<td>&quot;Bay&quot; probably refers to bay leaf; &quot;bring&quot; used in the context &quot;bring to a boil&quot;</td>
</tr>
<tr>
<td>12</td>
<td>Poultry</td>
<td>chicken, turkey, breast, poultr(y), stuf(fing), broth, casserol, skinless, bone, celer(y), sage, boneless, cook, mushroom, bird</td>
<td>Not sure why &quot;mushroom&quot; is categorised together with poultry.</td>
</tr>
<tr>
<td>13</td>
<td>East Asian</td>
<td>soy, sesam(e), sauc(e), chines(e), fr(y), oil, tofu, wok, tablespoon, ginger, sprout, chestnut, cornstarch, peanut, scallion</td>
<td> </td>
</tr>
<tr>
<td>14</td>
<td>Asian spices</td>
<td>spice, ginger, ground, allspic(e), cinnamon, clove, teaspoon, ml, raisin, chutney, seed, currant, cardamom, nutmeg, mace, store</td>
<td>Odd that teaspoon and ml are used to measure spices instead of grams. &quot;Store&quot; could refer to storage of spices, or could refer to the expectation that western readers need to specifically go to an Asian Store to get spices</td>
</tr>
<tr>
<td>15</td>
<td>Salad</td>
<td>salad, dress(ing), vinegar, cucumb(er), mustard, lettuc(e), pepper, tablespoon, green, red, mayonnais(e), toss, dijon, ingredi(ent), prepar(e)</td>
<td>&quot;Red&quot; and &quot;green&quot; probably refer to types of pepper</td>
</tr>
<tr>
<td>16</td>
<td>Dough (bread)</td>
<td>dough, yeast, roll, rise, knead, warm, let, make, water, place, ball, hand, doubl(e), flour, work</td>
<td> </td>
</tr>
<tr>
<td>17</td>
<td>Nutrition</td>
<td>g, fat, mg, calori(es), sodium, protein, cholesterol, carbohydr(ate), exchang(e), diabet(es), cal, gs, fiber, low, nutrit(ion)</td>
<td> </td>
</tr>
<tr>
<td>18</td>
<td>Chocolate</td>
<td>chocol(ate), cocoa, chip, cake, cand(y), vanilla, melt, browni(es), cream, frost, coffe(e), semisweet, sugar, marshmallow, cool</td>
<td>&quot;Cool&quot; probably refers to the common process of melting chocolate-based recipes and then letting it cool to set and harden</td>
</tr>
<tr>
<td>19</td>
<td>Legumes</td>
<td>bean, potato, water, mash, soak, drain, cook, black, garbanzo, tender, pea, kidney, ham, rins(e), pinto</td>
<td>Soaking or rinsing and then draining is common for legumes. Mashing is also common. &quot;Ham&quot; is probably on this list because some beans may commonly be eaten or cooked with strongly cured ham.</td>
</tr>
<tr>
<td>20</td>
<td>Pasta</td>
<td>shrimp, noodl(e), ricotta, lasagna, spachetti, sauc(e), devein, pasta, mozzarella, lasagn(a), min, cook, chees(e), ounce, cottag(e)</td>
<td>Pasta boiling time is measured in mins. Not sure why shrimp is on this list; don&#39;t expect it to be so strongly associated with pastas. &quot;Devein&quot; is a common procedure for shrimps.</td>
</tr>
<tr>
<td>21</td>
<td>Jelly</td>
<td>water, boil, gelatin, syrup, stir, sugar, saucepan, heat, dissolv(e), lemon, bring, cold, juic(e), mixtur(e), cornstarch, remov(e)</td>
<td>16th term is &quot;jelli&quot; (jelly). Some of these terms here probably refer to the common process of starting with cold water, bringing it to a boil, and then removing from heat while stirring.</td>
</tr>
<tr>
<td>22</td>
<td>Preservation</td>
<td>jar, pint, ft, canner, canning, pressu(re), process, pickl(e), tabl(e), lid, recommend, altitud(e), quart, pound, headspac(e)</td>
<td>Suspect that &quot;altitud(e)&quot; and &quot;headspac(e)&quot; are personal terms (e.g. &quot;taking time to can gives you headspace). 17th and 19th on the list are &quot;Steril(e)&quot; and &quot;USDA&quot;, which would be important for DIY canning.</td>
</tr>
<tr>
<td>23</td>
<td>Cheese (hot)</td>
<td>chees(e), cheddar, macaroni, broccoli, quich(e), shred, grate, bisquick, parmesan, cup, prepar(e), swiss, egg, zucchini, bake</td>
<td>The vegetables on this list are commonly used in quiche or casseroles (18th on the list). No idea why Bisquick is on this list; perhaps it&#39;s a common ingredient for quiches or casseroles as well.</td>
</tr>
<tr>
<td>24</td>
<td>Meat</td>
<td>meat, beef, sauc(e), teaspoon, pork, ground, worchestershir(e), meatbal(l), tablespoon, pepper, patt(y), marinad(e), garlic, barbecu€</td>
<td> </td>
</tr>
<tr>
<td>25</td>
<td>Blender</td>
<td>processor, food, blender, process, pure, chile, blade, smooth, puls(e), chipotl(e), tahini, tomatillo, seed, blend, garlic</td>
<td>&quot;Smooth&quot; probably refers to &quot;blend until smooth&quot;. &quot;Puls(e)&quot; would refer to &quot;pulsing a blender&quot;</td>
</tr>
<tr>
<td>26</td>
<td>Frying</td>
<td>mushroom, onion, pepper, saut, heat, add, skillet, oil, minut(e), cook, tomato, rice, chop, garlic, stir</td>
<td> Pepper could refer to ground peppercorns, or could refer to bell peppers (bell is 16th on the list). Both are commonly used in frying. Frying time is often listed in minutes (e.g. &quot;fry for 5 minutes, then add…&quot;) </td>
</tr>
<tr>
<td>27</td>
<td>Sweet dessert</td>
<td>pumpkin, peanut, cracker, pud(ding), mix, graham, milk, cup, condens(e), prepar(e), sweeten, jello, cereal, whip, key</td>
<td>&quot;Condens(e)&quot; should refer to condensed milk. &quot;key&quot; might refer to key lime.</td>
</tr>
<tr>
<td>28</td>
<td>Mediterranian</td>
<td>oliv(e), pasta, basil, oil, garlic, tomato, fresh, italian, parsley, pepper, herb, parmesan, pine, virgin, chop</td>
<td>&quot;Pine&quot; probably refers to pine nuts, used in pesto (17th on the list). &quot;virgin&quot; refers to the olive oil.</td>
</tr>
<tr>
<td>29</td>
<td>Cakes</td>
<td>cake, sugar, cream, beat, vanilla, egg, pan, cup, cheesecak(e), bake, teaspoon, butter, prepar(e), cool, flour</td>
<td>&quot;Cup&quot; could refer to cupcakes or to measuring cups. Most baked cakes need to be allowed time to cool.</td>
</tr>
<tr>
<td>30</td>
<td>Cookies</td>
<td>cooki(es), bake, flour, sheet, sugar, dough, teaspoon, butter, soda, roll, shorten, egg, brown, purpos(e), oven</td>
<td>&quot;Shorten&quot; probably refers to &quot;shortening&quot;, a common baking ingredient. &quot;Soda&quot; refers to &quot;baking soda&quot;. Presumably baking recipes may explain the &quot;purpos(e)&quot; of certain steps or ingredients.</td>
</tr>
<tr>
<td>31</td>
<td>Flatbreads</td>
<td>pizza, bread, chees(e), bake, oven, crumb, phyllo, slice, sprinkl(e), crepe, preheat, mozzarella, sheet, parmesan, artichok(e)</td>
<td>&quot;Phyllo&quot;, also known as &quot;filo&quot;, is a type of dough used in middle eastern cooking. &quot;Sprinkl(e)&quot; probably refers to sprinkling ingredients over a pizza or other flatbread</td>
</tr>
<tr>
<td>32</td>
<td>Fruit tarts</td>
<td>pie, orang(e), pineappl(e), banana, crust, peach, cherr(y), shell, fruit, juic(e), whip, unbak(ed), raspberr(y), blueberr(y), dessert</td>
<td>&quot;Crust&quot; and &quot;shell&quot; should be parts of the pie, not the fruit. &quot;Unbaked&quot; is likely part of a warning about taking it out of the oven too early. &quot;Whip&quot; may be &quot;whipped cream&quot;</td>
</tr>
<tr>
<td>33</td>
<td>Microwave</td>
<td>microwav(e), casserol(e), uncov(er), ounce, power, high, cn, minut(e), safe, dish, quart, stir, micro, oven, drain</td>
<td>Microwave instructions usually are given in minutes and instruct on what level of power to set. &quot;Safe&quot; may refer to safety instructions, such as &quot;remove when it has cooled down enough to handle safely&quot;.</td>
</tr>
<tr>
<td>34</td>
<td>Breads</td>
<td>bread, muffin, flour, bake, batter, loaf, egg, milk, teaspoon, powder, ingredi(ents), soda, buttermilk, wheat, salt</td>
<td> </td>
</tr>
<tr>
<td>35</td>
<td>Curry</td>
<td>corn, coriand(er), turmer(ic), kernel, indian, cumin, curr(y), seed, ghee, okra, masala, chilli, garam, pop, teaspoon</td>
<td> </td>
</tr>
</tbody></table>
</div>
Before we move on, I'd like to point out the various topics devoted to the subtle differences between various dough products and various fruit products! I'm finding this very fascinating. It seems like there are actually enough textual cues in the recipes for the algorithm to be able to differentiate between breads, flatbreads, cakes; chocolatey vs sugary vs savoury; and so on...  and as for fruits, you might noticed there's a topic with apples, pears and figs that I've called "dry fruits" for the want of a better word - that sounds like stuff you'd put together into a Christmas pie! 

And that's in contrast to the tart fruits you'd add to yogurt in the fresh fruits topic, or pumpkin and peanut with dairy products under 'sweet desserts', or the stuff you'd put in a fruit tart. In contrast, surprisingly, my model didn't manage to differentiate much between different kinds of meat, such as pork, beef, lamb, and so on. My guess is that this is probably due to the dataset being imbalanced: I have a massive number of recipes under baked goods, sweets, and fruit products, but relatively few meat-based recipes in the corpus.

Alright, so now that I have a list of topics, I can use these to make sense of clustering. As with my earlier attempt, I merged all documents within each category of recipes into a single long document. I then used gensim's LDA model to assign weights for each topic for each document. This can be visualised in a heatmap...

<div style="height: 400px; width: 630px; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/topic%20distributions.png">
</div>
<div class="fineprint">(the data has been logarithmically scaled to provide better visualisation in the heatmap, since values can vary by orders of magnitude)</div>

...but that's a bit too much info to easily take in! That said, it <b><i>is</i></b> possible to see some basic trends in that heatmap.

Anyway, I used these weights to perform heirarchical clustering to obtain this result:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/heirarchy%20of%20recipes%20with%20gensim%20topics.png">

That looks pretty good! Now let's see what topics contribute to each cluster. 

Based on the results of the heirarchical clustering, I'll define the following clusters:

<div style="height: 200px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Cluster</th>
<th title="Field #2">Elements of cluster</th>
</tr></thead>
<tbody><tr>
<td align="right">1</td>
<td>Fruits, grains &amp; veg: salad fruit; Fruits, grains &amp; veg: pickle; side-dishes relish; side-dishes vinegar; Fruits, grains &amp; veg: bean-salads; Fruits, grains &amp; veg: potato-salads; Fruits, grains &amp; veg: salad; Fruits, grains &amp; veg: salad pasta; side-dishes dressing </td>
</tr>
<tr>
<td align="right">2</td>
<td>ethnic asia chinese; ethnic asia filipino; ethnic asia hawaiian; ethnic asia indonesian; ethnic asia japanese; ethnic asia korean; ethnic asia singapore; ethnic asia thai; ethnic asia vietnamese; main-dishes meat seafood </td>
</tr>
<tr>
<td align="right">3</td>
<td>main-dishes burger; side-dishes marinade; side-dishes rub </td>
</tr>
<tr>
<td align="right">4</td>
<td>ethnic europe greek; ethnic europe italian; main-dishes pasta; main-dishes pizza; side-dishes oil; side-dishes pesto </td>
</tr>
<tr>
<td align="right">5</td>
<td>Fruits, grains &amp; veg: pilaf; Fruits, grains &amp; veg: rice </td>
</tr>
<tr>
<td align="right">6</td>
<td>Fruits, grains &amp; veg: stuffing; main-dishes crockpot; main-dishes meat poultry </td>
</tr>
<tr>
<td align="right">7</td>
<td>ethnic europe basque; ethnic europe spanish </td>
</tr>
<tr>
<td align="right">8</td>
<td>ethnic africa middle-east; ethnic africa middle-east armenian; ethnic africa middle-east lebanese; ethnic africa middle-east turkish; ethnic africa morocco </td>
</tr>
<tr>
<td align="right">9</td>
<td>main-dishes meat; side-dishes sauce </td>
</tr>
<tr>
<td align="right">10</td>
<td>ethnic africa; ethnic africa middle-east persian; ethnic america brazil; ethnic america cajun; ethnic america caribbean; ethnic america peruvian </td>
</tr>
<tr>
<td align="right">11</td>
<td>ethnic africa ethiopian; Fruits, grains &amp; veg: polenta; Fruits, grains &amp; veg: stuffed-veg; Fruits, grains &amp; veg: vegetable; special-diets vegetarian </td>
</tr>
<tr>
<td align="right">12</td>
<td>ethnic asia indian; side-dishes butter; side-dishes chutney; side-dishes condiment; side-dishes spice </td>
</tr>
<tr>
<td align="right">13</td>
<td>ethnic america mexican; side-dishes salsa; soup chili </td>
</tr>
<tr>
<td align="right">14</td>
<td>Fruits, grains &amp; veg: beans-grains; special-diets babyfood </td>
</tr>
<tr>
<td align="right">15</td>
<td>main-dishes sandwich; munchies appetizer; munchies dips-spreads </td>
</tr>
<tr>
<td align="right">16</td>
<td>main-dishes casserole; main-dishes dinner-pies; main-dishes egg; misc microwave; side-dishes cheese; special-diets diabetic </td>
</tr>
<tr>
<td align="right">17</td>
<td>ethnic america canadian; ethnic europe czech; ethnic europe hungarian; ethnic europe ukrainian; soup soup </td>
</tr>
<tr>
<td align="right">18</td>
<td>ethnic europe irish; ethnic europe scottish; ethnic europe welsh </td>
</tr>
<tr>
<td align="right">19</td>
<td>ethnic america native; main-dishes breakfast </td>
</tr>
<tr>
<td align="right">20</td>
<td>ethnic europe danish; ethnic europe french; ethnic europe norwegian </td>
</tr>
<tr>
<td align="right">21</td>
<td>ethnic europe finnish; ethnic europe polish; ethnic europe swedish </td>
</tr>
<tr>
<td align="right">22</td>
<td>ethnic europe british; ethnic europe german; ethnic europe russian; ethnic europe swiss; ethnic non-regional jewish </td>
</tr>
<tr>
<td align="right">23</td>
<td>ethnic asia australian; ethnic europe portuguese; misc medieval </td>
</tr>
<tr>
<td align="right">24</td>
<td>baked-goods dessert cobbler; baked-goods dessert pie; baked-goods dessert tart; dessert ; side-dishes pudding </td>
</tr>
<tr>
<td align="right">25</td>
<td>baked-goods pastry; ethnic europe austrian; holiday christmas </td>
</tr>
<tr>
<td align="right">26</td>
<td>holiday ; holiday halloween; misc camping; misc copycat; misc kid; misc mix; munchies snack </td>
</tr>
<tr>
<td align="right">27</td>
<td>baked-goods dessert cooky; baked-goods scone </td>
</tr>
<tr>
<td align="right">28</td>
<td>baked-goods bread; baked-goods bun; baked-goods muffin; baked-goods roll; misc pet-food dog; special-diets gluten-free </td>
</tr>
<tr>
<td align="right">29</td>
<td>baked-goods dessert cake; baked-goods dessert cheesecake </td>
</tr>
<tr>
<td align="right">30</td>
<td>dessert frozen-desserts; dessert mousse; dessert trifle </td>
</tr>
<tr>
<td align="right">31</td>
<td>dessert candy; dessert chocolate; holiday easter </td>
</tr>
<tr>
<td align="right">32</td>
<td>baked-goods dessert brownie; dessert frosting </td>
</tr>
<tr>
<td align="right">33</td>
<td>misc canning </td>
</tr>
<tr>
<td align="right">34</td>
<td>side-dishes jam; side-dishes preserve </td>
</tr>
<tr>
<td align="right">35</td>
<td>dessert frozen-yogurt; dessert sherbet; dessert sorbet </td>
</tr>
<tr>
<td align="right">36</td>
<td>Fruits, grains &amp; veg: fruit; side-dishes beverage </td>
</tr>
</tbody></table>
</div>

and then I calculated the means weights of each topic for the categories within each cluster. That's a mouthful, but basically what I'm saying is that for each cluster, I wanted to know, as a whole, which topics were most important overall. 

I'd think of these as the inter-cluster differentiating topics: they help to distinguish between clusters.

Here's the result presented in a heatmap (with more interpretation further down...)

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/topics%20of%20clusters.png">
<div class="fineprint">(again, the data for this heatmap and the next has been logarithmically scaled to provide better visualisation in the heatmap, since values can vary by orders of magnitude)</div>

In addition, I calculated the standard deviation of the weights for each topic for the categories within each cluster. In layman terms, I wanted to know, which each cluster, which topics had the most variation. So while the average weights would tell me which topics are most important for that cluster, the topics with the most variation for that cluster would tell me which topics are likely to be important for differentiating between the categories in that cluster.

I'd think of these as the <b><i>intra</i></b>-cluster differentiating topics: they help to distinguish between categories within each cluster.

Here's the result presented in a heatmap:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/variance%20of%20clusters.png">
<div class="fineprint">(data was logarithmically scaled; cluster 33 has no values because that cluster has only one category (canning), and therefore there is no variance whatsoever)</div>

This is a good start to get a quick idea of what topics might be important for each cluster! You might have realised that I'm doing something like dimensionality reduction again, this time with clustering, so that instead of having to look at all 150+ recipes, I'm looking at clusters of similar recipes.

<div class="fineprint">Strictly speaking, this isn't dimensionality reduction, since dimensionality is supposed to refer to the data itself (e.g. number of times each word appears) while the categories here are more like labels. But I'm using it with the same endgoal in mind: to make it easier to extract human-intuitive information by reducing the number of things to look at.</div>

We could get more insights by taking a look at this in table form. I set an arbitrary threshold and extracted the topics with the highest mean and largest variance for each cluster:

<div style="height: 200px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Cluster</th>
<th title="Field #2">Elements of cluster</th>
<th title="Field #3">Inter-cluster differentiating topics</th>
<th title="Field #4">Intra-cluster differentiating topics</th>
</tr></thead>
<tbody><tr>
<td align="right">1</td>
<td>Fruits, grains &amp; veg: salad fruit; Fruits, grains &amp; veg: pickle; side-dishes relish; side-dishes vinegar; Fruits, grains &amp; veg: bean-salads; Fruits, grains &amp; veg: potato-salads; Fruits, grains &amp; veg: salad; Fruits, grains &amp; veg: salad pasta; side-dishes dressing </td>
<td>Salad; Mediterranian; Legumes; Fresh fruits; Preservation; Jelly; Asian spices; Nutrition; Fruit tarts; Grilling</td>
<td>Legumes; Preservation; Mediterranian; Fresh fruits; Salad; Jelly; Fruit tarts; Asian spices; Dry fruits; Frying; Grilling; Blender; East Asian; Meat; Egg &amp; dairy; Poultry; Pasta; Curry; Cheese (hot); Nutrition; Grains; Dough (bread); Creamy spreads; Seafood; Mexican</td>
</tr>
<tr>
<td align="right">2</td>
<td>ethnic asia chinese; ethnic asia filipino; ethnic asia hawaiian; ethnic asia indonesian; ethnic asia japanese; ethnic asia korean; ethnic asia singapore; ethnic asia thai; ethnic asia vietnamese; main-dishes meat seafood </td>
<td>East Asian; Seafood; Dough (bread); Grilling; Meat; Frying; Salad; Soup stock; Asian spices; Dough handling; Poultry; Grains; Pasta</td>
<td>East Asian; Seafood; Meat; Frying; Dough (bread); Fruit tarts; Soup stock; Grilling; Salad; Asian spices; Slow cooking; Pasta; Egg &amp; dairy; Dough handling; Breads; Mediterranian; Curry; Poultry; Fresh fruits; Flatbreads; Jelly; Grains; Blender; Cakes</td>
</tr>
<tr>
<td align="right">3</td>
<td>main-dishes burger; side-dishes marinade; side-dishes rub </td>
<td>Meat; Asian spices; Grilling; Dough (bread); Blender; Salad; Seafood; East Asian; Poultry; Frying; Fresh fruits; Creamy spreads; Nutrition</td>
<td>Meat; Asian spices; Grilling; Seafood; Dough (bread); Frying; Creamy spreads; East Asian; Blender; Flatbreads; Salad; Fresh fruits; Mediterranian; Nutrition; Mexican; Poultry; Microwave; Slow cooking</td>
</tr>
<tr>
<td align="right">4</td>
<td>ethnic europe greek; ethnic europe italian; main-dishes pasta; main-dishes pizza; side-dishes oil; side-dishes pesto </td>
<td>Mediterranian; Flatbreads; Dough (bread); Blender; Frying; Pasta; Cheese (hot); Grilling; Egg &amp; dairy; Dough handling; Soup stock; East Asian; Grains; Seafood; Asian spices; Meat; Jelly</td>
<td>Mediterranian; Flatbreads; Blender; Pasta; Dough (bread); Frying; East Asian; Asian spices; Jelly; Egg &amp; dairy; Cheese (hot); Dough handling; Soup stock; Grilling; Grains; Salad; Fresh fruits; Seafood; Cookies; Mexican; Cakes; Meat; Preservation; Curry; Nutrition; Chocolate</td>
</tr>
<tr>
<td align="right">5</td>
<td>Fruits, grains &amp; veg: pilaf; Fruits, grains &amp; veg: rice </td>
<td>Grains; Frying; Asian spices; Mediterranian; Nutrition; East Asian; Soup stock; Curry; Poultry; Dough handling</td>
<td>Grains; East Asian; Asian spices; Nutrition; Frying; Seafood; Cheese (hot); Curry; Mexican</td>
</tr>
<tr>
<td align="right">6</td>
<td>Fruits, grains &amp; veg: stuffing; main-dishes crockpot; main-dishes meat poultry </td>
<td>Poultry; Slow cooking; Frying; Meat; Soup stock; Breads; Grilling; Grains; Dough (bread); East Asian; Egg &amp; dairy; Legumes</td>
<td>Slow cooking; Poultry; Frying; Meat; Breads; Grilling; East Asian; Legumes; Egg &amp; dairy; Jelly; Grains; Soup stock; Seafood; Dough (bread); Cheese (hot); Dry fruits; Flatbreads; Mexican; Nutrition; Asian spices; Salad; Curry</td>
</tr>
<tr>
<td align="right">7</td>
<td>ethnic europe basque; ethnic europe spanish </td>
<td>Frying; Seafood; Grilling; Legumes; Soup stock; Egg &amp; dairy; Poultry; Flatbreads; Dough handling</td>
<td>Grilling; Legumes; Poultry; Egg &amp; dairy; Microwave; Slow cooking; Seafood; Pasta; Salad; Mediterranian; Flatbreads; Nutrition; Grains; Dough (bread); Frying; Soup stock; East Asian; Blender</td>
</tr>
<tr>
<td align="right">8</td>
<td>ethnic africa middle-east; ethnic africa middle-east armenian; ethnic africa middle-east lebanese; ethnic africa middle-east turkish; ethnic africa morocco </td>
<td>Grilling; Frying; Asian spices; Dough (bread); Grains; Mediterranian; Jelly; Soup stock; Salad; Dough handling; Cookies; Fresh fruits; Poultry; Meat; Legumes; Curry; Egg &amp; dairy; Blender</td>
<td>Grilling; Asian spices; Mediterranian; Soup stock; Salad; Meat; Cookies; Frying; Grains; Dough (bread); Egg &amp; dairy; Fresh fruits; Seafood; Dough handling; Blender; Legumes; Poultry; Jelly; Creamy spreads; Curry; Cakes</td>
</tr>
<tr>
<td align="right">9</td>
<td>main-dishes meat; side-dishes sauce </td>
<td>Meat; Frying; Mediterranian; Grilling; Egg &amp; dairy; Grains; Jelly; Blender; Seafood; Poultry; Soup stock; Dough (bread); East Asian; Slow cooking; Mexican; Asian spices; Salad</td>
<td>Grilling; Jelly; Mediterranian; Blender; Frying; Poultry; Grains; Egg &amp; dairy; Dough (bread); Dough handling; Chocolate; Legumes; Seafood; Soup stock; Asian spices; Salad; Meat; Breads; Fresh fruits; East Asian</td>
</tr>
<tr>
<td align="right">10</td>
<td>ethnic africa; ethnic africa middle-east persian; ethnic america brazil; ethnic america cajun; ethnic america caribbean; ethnic america peruvian </td>
<td>Frying; Meat; Seafood; Legumes; Soup stock; Grilling; Grains; Dough (bread); Asian spices; Poultry; Curry; Slow cooking; Mexican; Dough handling; Salad; Egg &amp; dairy; Jelly</td>
<td>Slow cooking; Mexican; Grilling; Frying; Seafood; Asian spices; Curry; Soup stock; Meat; Legumes; Grains; Fresh fruits; Egg &amp; dairy; Jelly; Cookies; Dry fruits; Dough (bread); Poultry; Blender; Flatbreads; Mediterranian; Breads; Salad; Cheese (hot); Dough handling</td>
</tr>
<tr>
<td align="right">11</td>
<td>ethnic africa ethiopian; Fruits, grains &amp; veg: polenta; Fruits, grains &amp; veg: stuffed-veg; Fruits, grains &amp; veg: vegetable; special-diets vegetarian </td>
<td>Frying; Grains; Mediterranian; Asian spices; Grilling; Flatbreads; Curry; Legumes; East Asian; Dough (bread); Dough handling; Breads; Soup stock; Cheese (hot); Salad; Nutrition; Blender; Meat; Egg &amp; dairy</td>
<td>Asian spices; Frying; East Asian; Grilling; Mediterranian; Dough (bread); Legumes; Dough handling; Salad; Jelly; Curry; Breads; Flatbreads; Soup stock; Grains; Meat; Poultry; Cheese (hot); Microwave; Egg &amp; dairy; Nutrition; Mexican; Blender</td>
</tr>
<tr>
<td align="right">12</td>
<td>ethnic asia indian; side-dishes butter; side-dishes chutney; side-dishes condiment; side-dishes spice </td>
<td>Asian spices; Curry; Meat; Jelly; Blender; Salad; Dry fruits; Preservation; Frying; Fresh fruits; Grains; Dough (bread); Seafood; Egg &amp; dairy</td>
<td>Curry; Asian spices; Meat; Jelly; Salad; Dry fruits; Frying; Blender; Preservation; Grains; Dough (bread); Egg &amp; dairy; Fresh fruits; Grilling; Sweet dessert; Mexican; Mediterranian; Creamy spreads; Poultry; Legumes; Fruit tarts</td>
</tr>
<tr>
<td align="right">13</td>
<td>ethnic america mexican; side-dishes salsa; soup chili </td>
<td>Mexican; Blender; Frying; Meat; Salad; Legumes; Grilling; Soup stock; Dough (bread); Fresh fruits; Poultry; Mediterranian</td>
<td>Blender; Salad; Meat; Frying; Legumes; Grilling; Mexican; Fresh fruits; Soup stock; Dough (bread); Mediterranian; Egg &amp; dairy; Poultry; Grains; Dough handling; Slow cooking; Cookies; Seafood; Cheese (hot); Microwave</td>
</tr>
<tr>
<td align="right">14</td>
<td>Fruits, grains &amp; veg: beans-grains; special-diets babyfood </td>
<td>Legumes; Egg &amp; dairy; Dough (bread); Grains; Frying; Blender; Breads; Poultry; Fresh fruits; Asian spices; Cheese (hot); Meat</td>
<td>Egg &amp; dairy; Frying; Dough (bread); Grains; Blender; Poultry; Asian spices; Breads; Meat; Cheese (hot); Mexican; Mediterranian; Fruit tarts; Curry; Cookies; Salad; Fresh fruits; Soup stock; Microwave; Nutrition; East Asian; Slow cooking</td>
</tr>
<tr>
<td align="right">15</td>
<td>main-dishes sandwich; munchies appetizer; munchies dips-spreads </td>
<td>Creamy spreads; Grilling; Meat; Flatbreads; Salad; Frying; Mediterranian; Mexican; Blender; East Asian; Dough (bread); Dough handling; Fresh fruits; Nutrition; Breads; Legumes; Egg &amp; dairy</td>
<td>Creamy spreads; Grilling; Meat; Blender; Flatbreads; Frying; Dough handling; Salad; Mexican; Fresh fruits; Dough (bread); Mediterranian; East Asian; Egg &amp; dairy; Poultry; Legumes; Breads; Cookies</td>
</tr>
<tr>
<td align="right">16</td>
<td>main-dishes casserole; main-dishes dinner-pies; main-dishes egg; misc microwave; side-dishes cheese; special-diets diabetic </td>
<td>Cheese (hot); Egg &amp; dairy; Microwave; Frying; Nutrition; Creamy spreads; Flatbreads; Breads; Mediterranian; Dough (bread); Dough handling; Meat; Grilling; Poultry; Legumes; Mexican; Salad; Grains</td>
<td>Cheese (hot); Microwave; Nutrition; Egg &amp; dairy; Creamy spreads; Frying; Poultry; Dough (bread); Salad; Flatbreads; Grilling; Grains; Mediterranian; Jelly; Chocolate; Cookies; Breads; Dough handling; Mexican; Fruit tarts; Legumes; Soup stock; Pasta; Sweet dessert; Meat; East Asian; Blender; Fresh fruits; Dry fruits</td>
</tr>
<tr>
<td align="right">17</td>
<td>ethnic america canadian; ethnic europe czech; ethnic europe hungarian; ethnic europe ukrainian; soup soup </td>
<td>Soup stock; Dough (bread); Egg &amp; dairy; Cookies; Meat; Frying; Grains; Slow cooking; Legumes; Cakes; Dry fruits; Asian spices; Breads; Grilling</td>
<td>Soup stock; Dough (bread); Cookies; Cakes; Dry fruits; Frying; Meat; Grains; Egg &amp; dairy; Asian spices; Slow cooking; Grilling; Mediterranian; Chocolate; Breads; Seafood; Jelly; Flatbreads; Blender; Legumes; Pasta; Curry; Fruit tarts; Creamy spreads; Cheese (hot); East Asian; Mexican</td>
</tr>
<tr>
<td align="right">18</td>
<td>ethnic europe irish; ethnic europe scottish; ethnic europe welsh </td>
<td>Breads; Soup stock; Dough (bread); Cookies; Asian spices; Cakes; Egg &amp; dairy; Dough handling; Grilling; Meat; Slow cooking; Legumes; Dry fruits; Nutrition; Jelly</td>
<td>Cookies; Soup stock; Jelly; Nutrition; Egg &amp; dairy; Legumes; Cakes; Asian spices; Dough handling; Slow cooking; Microwave; Grilling; Dough (bread); Dry fruits; Breads; Chocolate; Grains; Cheese (hot); Meat; Sweet dessert; Poultry</td>
</tr>
<tr>
<td align="right">19</td>
<td>ethnic america native; main-dishes breakfast </td>
<td>Breads; Jelly; Egg &amp; dairy; Legumes; Curry; Cheese (hot); Dough (bread); Soup stock; Frying; Mexican; Salad; Grilling; Flatbreads; Dry fruits; Nutrition; Cookies; Asian spices</td>
<td>Cheese (hot); Curry; Soup stock; Jelly; Legumes; Salad; Egg &amp; dairy; Flatbreads; Dry fruits; Fruit tarts; Fresh fruits; Cakes; Seafood; Creamy spreads; Nutrition; Asian spices; Sweet dessert; Mexican; Breads; Blender</td>
</tr>
<tr>
<td align="right">20</td>
<td>ethnic europe danish; ethnic europe french; ethnic europe norwegian </td>
<td>Egg &amp; dairy; Dough (bread); Soup stock; Seafood; Grilling; Cookies; Breads; Creamy spreads; Salad; Dough handling; Cakes; Poultry; Legumes; Meat; Frying; Chocolate; Jelly</td>
<td>Soup stock; Creamy spreads; Frying; Cookies; Poultry; Grilling; Dough handling; Jelly; Chocolate; Seafood; Cakes; Mediterranian; Dry fruits; Grains; Salad; Meat; Flatbreads; Breads; Dough (bread); Legumes</td>
</tr>
<tr>
<td align="right">21</td>
<td>ethnic europe finnish; ethnic europe polish; ethnic europe swedish </td>
<td>Dough (bread); Soup stock; Breads; Egg &amp; dairy; Cakes; Slow cooking; Frying; Meat; Legumes; Seafood; Asian spices; Jelly; Grilling; Dry fruits; Fresh fruits</td>
<td>Slow cooking; Breads; Soup stock; Dough (bread); Frying; Grilling; Meat; Flatbreads; Cakes; Dry fruits; Asian spices; Seafood; Legumes; Cookies; Fresh fruits; Salad; Curry; Chocolate; Creamy spreads; Microwave</td>
</tr>
<tr>
<td align="right">22</td>
<td>ethnic europe british; ethnic europe german; ethnic europe russian; ethnic europe swiss; ethnic non-regional jewish </td>
<td>Dough (bread); Soup stock; Breads; Egg &amp; dairy; Cakes; Asian spices; Cookies; Grilling; Dough handling; Frying; Jelly; Meat; Dry fruits; Cheese (hot); Creamy spreads; Legumes</td>
<td>Meat; Frying; Cakes; Cheese (hot); Dough (bread); Breads; Soup stock; Asian spices; Egg &amp; dairy; Nutrition; Grilling; Cookies; Dough handling; Salad; Creamy spreads; Dry fruits; Chocolate; Poultry; Jelly; Pasta; Slow cooking; Seafood; Fresh fruits</td>
</tr>
<tr>
<td align="right">23</td>
<td>ethnic asia australian; ethnic europe portuguese; misc medieval </td>
<td>Dough (bread); Seafood; Asian spices; Meat; Breads; Soup stock; Grilling; Egg &amp; dairy; Frying; Cakes; Legumes; Poultry; Dough handling; Nutrition; Mediterranian; Dry fruits</td>
<td>Asian spices; Frying; Legumes; Poultry; Seafood; Cakes; Dough handling; Dry fruits; Dough (bread); Egg &amp; dairy; Grilling; Jelly; Breads; Grains; Nutrition; Soup stock; Cookies; Meat; Fruit tarts; East Asian; Flatbreads; Mediterranian</td>
</tr>
<tr>
<td align="right">24</td>
<td>baked-goods dessert cobbler; baked-goods dessert pie; baked-goods dessert tart; dessert ; side-dishes pudding </td>
<td>Cakes; Egg &amp; dairy; Cookies; Dry fruits; Fruit tarts; Jelly; Chocolate; Dough handling; Breads; Sweet dessert; Fresh fruits; Dough (bread)</td>
<td>Cookies; Fruit tarts; Breads; Egg &amp; dairy; Dough handling; Chocolate; Cakes; Dry fruits; Sweet dessert; Fresh fruits; Grains; Dough (bread); Asian spices; Jelly</td>
</tr>
<tr>
<td align="right">25</td>
<td>baked-goods pastry; ethnic europe austrian; holiday christmas </td>
<td>Dough (bread); Cookies; Cakes; Egg &amp; dairy; Chocolate; Breads; Dough handling; Asian spices; Dry fruits</td>
<td>Cookies; Egg &amp; dairy; Breads; Chocolate; Dough handling; Cakes; Dry fruits; Asian spices; Soup stock; Grilling; Jelly; Dough (bread); Fruit tarts; Sweet dessert; Salad; Fresh fruits; Meat</td>
</tr>
<tr>
<td align="right">26</td>
<td>holiday ; holiday halloween; misc camping; misc copycat; misc kid; misc mix; munchies snack </td>
<td>Dough (bread); Sweet dessert; Cookies; Chocolate; Breads; Jelly; Dough handling; Meat; Grains; Egg &amp; dairy; Legumes; Salad; Asian spices; Microwave; Mexican; Flatbreads; Creamy spreads</td>
<td>Sweet dessert; Dough (bread); Dough handling; Chocolate; Breads; Meat; Cookies; Grains; Egg &amp; dairy; Jelly; Mexican; Legumes; Microwave; Grilling; Asian spices; Curry; Cakes; Slow cooking; Flatbreads; Poultry; Salad; Soup stock; Dry fruits; Fresh fruits; Seafood; Fruit tarts; Creamy spreads; Cheese (hot)</td>
</tr>
<tr>
<td align="right">27</td>
<td>baked-goods dessert cooky; baked-goods scone </td>
<td>Cookies; Breads; Chocolate; Dough (bread); Cakes; Fresh fruits; Nutrition; Dry fruits; Dough handling</td>
<td>Breads; Chocolate; Cookies; Cakes; Dry fruits; Fresh fruits; Sweet dessert; Dough (bread); Nutrition; Flatbreads; Dough handling</td>
</tr>
<tr>
<td align="right">28</td>
<td>baked-goods bread; baked-goods bun; baked-goods muffin; baked-goods roll; misc pet-food dog; special-diets gluten-free </td>
<td>Breads; Dough (bread); Cookies; Cakes; Nutrition; Flatbreads; Grains</td>
<td>Breads; Dough (bread); Cookies; Nutrition; Cakes; Flatbreads; Meat; Poultry; Grains; Dry fruits; Fruit tarts; Legumes; Fresh fruits; Asian spices; Jelly; Chocolate</td>
</tr>
<tr>
<td align="right">29</td>
<td>baked-goods dessert cake; baked-goods dessert cheesecake </td>
<td>Cakes; Chocolate; Breads; Jelly; Sweet dessert; Dough (bread); Fruit tarts; Fresh fruits; Dough handling; Dry fruits</td>
<td>Cakes; Breads; Cookies; Sweet dessert; Asian spices; Dry fruits; Chocolate; Egg &amp; dairy; Dough (bread); Jelly; Blender</td>
</tr>
<tr>
<td align="right">30</td>
<td>dessert frozen-desserts; dessert mousse; dessert trifle </td>
<td>Chocolate; Egg &amp; dairy; Jelly; Fresh fruits; Cakes; Sweet dessert; Fruit tarts; Dough (bread); Dough handling</td>
<td>Chocolate; Jelly; Egg &amp; dairy; Fresh fruits; Sweet dessert; Dough handling; Fruit tarts; Dough (bread); Blender</td>
</tr>
<tr>
<td align="right">31</td>
<td>dessert candy; dessert chocolate; holiday easter </td>
<td>Chocolate; Dough (bread); Sweet dessert; Jelly; Cookies; Cakes; Egg &amp; dairy; Dough handling; Breads</td>
<td>Jelly; Chocolate; Sweet dessert; Cakes; Dough (bread); Cookies; Egg &amp; dairy; Breads; Dough handling; Fresh fruits</td>
</tr>
<tr>
<td align="right">32</td>
<td>baked-goods dessert brownie; dessert frosting </td>
<td>Chocolate; Cakes; Breads; Jelly; Cookies; Egg &amp; dairy; Nutrition</td>
<td>Breads; Jelly; Chocolate; Cakes; Cookies; Egg &amp; dairy; Dough (bread); Fresh fruits; Grains</td>
</tr>
<tr>
<td align="right">33</td>
<td>misc canning </td>
<td>Preservation; Jelly; Dough (bread); Dry fruits; Legumes</td>
<td>n/a (only one category of recipes in this cluster)</td>
</tr>
<tr>
<td align="right">34</td>
<td>side-dishes jam; side-dishes preserve </td>
<td>Jelly; Preservation; Fresh fruits; Asian spices; Dry fruits; Sweet dessert; Fruit tarts; Dough (bread); Blender</td>
<td>Jelly; Asian spices; Sweet dessert; Dry fruits; Preservation; Mediterranian; Egg &amp; dairy; Soup stock; Microwave; Fresh fruits; Slow cooking</td>
</tr>
<tr>
<td align="right">35</td>
<td>dessert frozen-yogurt; dessert sherbet; dessert sorbet </td>
<td>Fresh fruits; Jelly; Blender; Fruit tarts; Chocolate; Cakes; Nutrition; Sweet dessert; Dry fruits</td>
<td>Jelly; Chocolate; Sweet dessert; Fruit tarts; Nutrition; Cakes; Fresh fruits; Blender; Cheese (hot); Egg &amp; dairy; Dry fruits; Dough handling; Asian spices</td>
</tr>
<tr>
<td align="right">36</td>
<td>Fruits, grains &amp; veg: fruit; side-dishes beverage </td>
<td>Fresh fruits; Jelly; Dry fruits; Fruit tarts; Asian spices; Chocolate; Nutrition; Dough (bread); Dough handling; Egg &amp; dairy; Preservation</td>
<td>Dry fruits; Fresh fruits; Chocolate; Asian spices; Dough handling; Fruit tarts; Grilling; Nutrition; Sweet dessert; Egg &amp; dairy; Grains; Mediterranian; Jelly; Salad; Dough (bread)</td>
</tr>
</tbody></table>
</div>

In this table, topics are arranged in order of descending importance. For example, in cluster 1, the topic of salad is the most important, followed by mediterranian, and then legumes. I cut it off at grilling, but there are additional topics after that with lower and lower importance.

Most of the time, the inter-cluster differentiating topics are easy to interpret. Of course the vegetable-based cluster would have a lot of discussion of salad, mediterranian food, and legumes! 

The intra-cluster differentiating topics are trickier, though. Most of the topics listed here also seem to be the topics with the highest means. My guess is that this is probably because topics with larger means will also be  topics with larger values, and therefore their standard deviation would be higher as well. When we're looking at orders of magnitude differences between topic weights, this would be quite substantial.

Therefore, I tried to scale my intra-cluster differentiator metric: instead of just using standard deviation, I used standard deviation divided by mean. This is illustrated in the following heatmap...

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/variance%20of%20clusters%202.png">

and after setting a new cutoff, I've identified a new set of intra-cluster differentiating topics:

<div style="height: 400px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Cluster</th>
<th title="Field #2">Elements</th>
<th title="Field #3">Inter-cluster differentiating topics</th>
<th title="Field #4">Intra-cluster differentiating topics</th>
</tr></thead>
<tbody><tr>
<td align="right">1</td>
<td>Fruits, grains &amp; veg: salad fruit; Fruits, grains &amp; veg: pickle; side-dishes relish; side-dishes vinegar; Fruits, grains &amp; veg: bean-salads; Fruits, grains &amp; veg: potato-salads; Fruits, grains &amp; veg: salad; Fruits, grains &amp; veg: salad pasta; side-dishes dressing </td>
<td>Salad; Mediterranian; Legumes; Fresh fruits; Preservation; Jelly; Asian spices; Nutrition; Fruit tarts; Grilling</td>
<td>Chocolate; Cookies; Fruit tarts; Cakes; Pasta; Flatbreads; Dry fruits; Meat; Frying; Preservation; Cheese (hot); Legumes; Soup stock; Egg &amp; dairy</td>
</tr>
<tr>
<td align="right">2</td>
<td>ethnic asia chinese; ethnic asia filipino; ethnic asia hawaiian; ethnic asia indonesian; ethnic asia japanese; ethnic asia korean; ethnic asia singapore; ethnic asia thai; ethnic asia vietnamese; main-dishes meat seafood </td>
<td>East Asian; Seafood; Dough (bread); Grilling; Meat; Frying; Salad; Soup stock; Asian spices; Dough handling; Poultry; Grains; Pasta</td>
<td>Flatbreads; Mediterranian; Fruit tarts; Cheese (hot); Cakes; Fresh fruits; Sweet dessert; Preservation; Chocolate; Creamy spreads; Slow cooking; Cookies</td>
</tr>
<tr>
<td align="right">3</td>
<td>main-dishes burger; side-dishes marinade; side-dishes rub </td>
<td>Meat; Asian spices; Grilling; Dough (bread); Blender; Salad; Seafood; East Asian; Poultry; Frying; Fresh fruits; Creamy spreads; Nutrition</td>
<td>Frying; Creamy spreads; Flatbreads; Microwave; Grains; Cookies; Breads; Dough handling; Soup stock; Jelly; Sweet dessert; Seafood; Mediterranian</td>
</tr>
<tr>
<td align="right">4</td>
<td>ethnic europe greek; ethnic europe italian; main-dishes pasta; main-dishes pizza; side-dishes oil; side-dishes pesto </td>
<td>Mediterranian; Flatbreads; Dough (bread); Blender; Frying; Pasta; Cheese (hot); Grilling; Egg &amp; dairy; Dough handling; Soup stock; East Asian; Grains; Seafood; Asian spices; Meat; Jelly</td>
<td>Chocolate; Curry; Preservation; East Asian; Pasta; Mexican; Flatbreads; Asian spices; Blender; Jelly; Fruit tarts</td>
</tr>
<tr>
<td align="right">5</td>
<td>Fruits, grains &amp; veg: pilaf; Fruits, grains &amp; veg: rice </td>
<td>Grains; Frying; Asian spices; Mediterranian; Nutrition; East Asian; Soup stock; Curry; Poultry; Dough handling</td>
<td>Cheese (hot); Preservation; Cakes; Meat; Breads; Chocolate; Seafood</td>
</tr>
<tr>
<td align="right">6</td>
<td>Fruits, grains &amp; veg: stuffing; main-dishes crockpot; main-dishes meat poultry </td>
<td>Poultry; Slow cooking; Frying; Meat; Soup stock; Breads; Grilling; Grains; Dough (bread); East Asian; Egg &amp; dairy; Legumes</td>
<td>Jelly; Slow cooking; Seafood; East Asian; Breads</td>
</tr>
<tr>
<td align="right">7</td>
<td>ethnic europe basque; ethnic europe spanish </td>
<td>Frying; Seafood; Grilling; Legumes; Soup stock; Egg &amp; dairy; Poultry; Flatbreads; Dough handling</td>
<td>Poultry; Dry fruits; Microwave; Pasta; Nutrition; Grains; Dough (bread); East Asian; Meat; Asian spices; Jelly; Mexican; Breads; Preservation</td>
</tr>
<tr>
<td align="right">8</td>
<td>ethnic africa middle-east; ethnic africa middle-east armenian; ethnic africa middle-east lebanese; ethnic africa middle-east turkish; ethnic africa morocco </td>
<td>Grilling; Frying; Asian spices; Dough (bread); Grains; Mediterranian; Jelly; Soup stock; Salad; Dough handling; Cookies; Fresh fruits; Poultry; Meat; Legumes; Curry; Egg &amp; dairy; Blender</td>
<td>Cheese (hot); Creamy spreads; Slow cooking; Fruit tarts; Chocolate</td>
</tr>
<tr>
<td align="right">9</td>
<td>main-dishes meat; side-dishes sauce </td>
<td>Meat; Frying; Mediterranian; Grilling; Egg &amp; dairy; Grains; Jelly; Blender; Seafood; Poultry; Soup stock; Dough (bread); East Asian; Slow cooking; Mexican; Asian spices; Salad</td>
<td>Dough handling; Cookies; Chocolate; Grilling; Legumes</td>
</tr>
<tr>
<td align="right">10</td>
<td>ethnic africa; ethnic africa middle-east persian; ethnic america brazil; ethnic america cajun; ethnic america caribbean; ethnic america peruvian </td>
<td>Frying; Meat; Seafood; Legumes; Soup stock; Grilling; Grains; Dough (bread); Asian spices; Poultry; Curry; Slow cooking; Mexican; Dough handling; Salad; Egg &amp; dairy; Jelly</td>
<td>Cheese (hot); Dry fruits; Mexican; Cakes; Pasta; Microwave; Creamy spreads; Fresh fruits; Slow cooking; Cookies; Preservation; Chocolate</td>
</tr>
<tr>
<td align="right">11</td>
<td>ethnic africa ethiopian; Fruits, grains &amp; veg: polenta; Fruits, grains &amp; veg: stuffed-veg; Fruits, grains &amp; veg: vegetable; special-diets vegetarian </td>
<td>Frying; Grains; Mediterranian; Asian spices; Grilling; Flatbreads; Curry; Legumes; East Asian; Dough (bread); Dough handling; Breads; Soup stock; Cheese (hot); Salad; Nutrition; Blender; Meat; Egg &amp; dairy</td>
<td>Jelly; Cakes; Asian spices; Cookies; Poultry; Chocolate; Seafood</td>
</tr>
<tr>
<td align="right">12</td>
<td>ethnic asia indian; side-dishes butter; side-dishes chutney; side-dishes condiment; side-dishes spice </td>
<td>Asian spices; Curry; Meat; Jelly; Blender; Salad; Dry fruits; Preservation; Frying; Fresh fruits; Grains; Dough (bread); Seafood; Egg &amp; dairy</td>
<td>Grilling; Cheese (hot); Curry; Chocolate; Dry fruits</td>
</tr>
<tr>
<td align="right">13</td>
<td>ethnic america mexican; side-dishes salsa; soup chili </td>
<td>Mexican; Blender; Frying; Meat; Salad; Legumes; Grilling; Soup stock; Dough (bread); Fresh fruits; Poultry; Mediterranian</td>
<td>Cakes; Cookies; Egg &amp; dairy; Creamy spreads; Flatbreads; Dough handling; Salad; Fresh fruits; Slow cooking; Breads</td>
</tr>
<tr>
<td align="right">14</td>
<td>Fruits, grains &amp; veg: beans-grains; special-diets babyfood </td>
<td>Legumes; Egg &amp; dairy; Dough (bread); Grains; Frying; Blender; Breads; Poultry; Fresh fruits; Asian spices; Cheese (hot); Meat</td>
<td>Frying; Asian spices; Meat; Mexican; Mediterranian; Curry; Salad; Soup stock; Microwave; Egg &amp; dairy; East Asian; Slow cooking; Flatbreads; Jelly; Preservation; Dough handling; Cakes; Creamy spreads; Cookies; Sweet dessert</td>
</tr>
<tr>
<td align="right">15</td>
<td>main-dishes sandwich; munchies appetizer; munchies dips-spreads </td>
<td>Creamy spreads; Grilling; Meat; Flatbreads; Salad; Frying; Mediterranian; Mexican; Blender; East Asian; Dough (bread); Dough handling; Fresh fruits; Nutrition; Breads; Legumes; Egg &amp; dairy</td>
<td>Chocolate; Cakes; Cookies</td>
</tr>
<tr>
<td align="right">16</td>
<td>main-dishes casserole; main-dishes dinner-pies; main-dishes egg; misc microwave; side-dishes cheese; special-diets diabetic </td>
<td>Cheese (hot); Egg &amp; dairy; Microwave; Frying; Nutrition; Creamy spreads; Flatbreads; Breads; Mediterranian; Dough (bread); Dough handling; Meat; Grilling; Poultry; Legumes; Mexican; Salad; Grains</td>
<td>Nutrition; Chocolate</td>
</tr>
<tr>
<td align="right">17</td>
<td>ethnic america canadian; ethnic europe czech; ethnic europe hungarian; ethnic europe ukrainian; soup soup </td>
<td>Soup stock; Dough (bread); Egg &amp; dairy; Cookies; Meat; Frying; Grains; Slow cooking; Legumes; Cakes; Dry fruits; Asian spices; Breads; Grilling</td>
<td>Mexican; Mediterranian; Fruit tarts; Pasta; East Asian; Flatbreads; Cakes; Sweet dessert; Chocolate; Preservation</td>
</tr>
<tr>
<td align="right">18</td>
<td>ethnic europe irish; ethnic europe scottish; ethnic europe welsh </td>
<td>Breads; Soup stock; Dough (bread); Cookies; Asian spices; Cakes; Egg &amp; dairy; Dough handling; Grilling; Meat; Slow cooking; Legumes; Dry fruits; Nutrition; Jelly</td>
<td>Chocolate; Frying; Curry; Pasta; East Asian; Jelly; Nutrition; Flatbreads</td>
</tr>
<tr>
<td align="right">19</td>
<td>ethnic america native; main-dishes breakfast </td>
<td>Breads; Jelly; Egg &amp; dairy; Legumes; Curry; Cheese (hot); Dough (bread); Soup stock; Frying; Mexican; Salad; Grilling; Flatbreads; Dry fruits; Nutrition; Cookies; Asian spices</td>
<td>Seafood; Flatbreads; Dry fruits; Fruit tarts; Fresh fruits; Cakes; Creamy spreads; Poultry; Salad; Slow cooking; Preservation; Pasta; Soup stock; Cheese (hot)</td>
</tr>
<tr>
<td align="right">20</td>
<td>ethnic europe danish; ethnic europe french; ethnic europe norwegian </td>
<td>Egg &amp; dairy; Dough (bread); Soup stock; Seafood; Grilling; Cookies; Breads; Creamy spreads; Salad; Dough handling; Cakes; Poultry; Legumes; Meat; Frying; Chocolate; Jelly</td>
<td>Mediterranian; Pasta; Preservation; Sweet dessert; Grains; Frying</td>
</tr>
<tr>
<td align="right">21</td>
<td>ethnic europe finnish; ethnic europe polish; ethnic europe swedish </td>
<td>Dough (bread); Soup stock; Breads; Egg &amp; dairy; Cakes; Slow cooking; Frying; Meat; Legumes; Seafood; Asian spices; Jelly; Grilling; Dry fruits; Fresh fruits</td>
<td>Curry; Poultry; Chocolate; Mexican; Blender; Pasta; Sweet dessert; Mediterranian; Slow cooking; Preservation; Grains</td>
</tr>
<tr>
<td align="right">22</td>
<td>ethnic europe british; ethnic europe german; ethnic europe russian; ethnic europe swiss; ethnic non-regional jewish </td>
<td>Dough (bread); Soup stock; Breads; Egg &amp; dairy; Cakes; Asian spices; Cookies; Grilling; Dough handling; Frying; Jelly; Meat; Dry fruits; Cheese (hot); Creamy spreads; Legumes</td>
<td>Mexican; Pasta; Curry; Microwave</td>
</tr>
<tr>
<td align="right">23</td>
<td>ethnic asia australian; ethnic europe portuguese; misc medieval </td>
<td>Dough (bread); Seafood; Asian spices; Meat; Breads; Soup stock; Grilling; Egg &amp; dairy; Frying; Cakes; Legumes; Poultry; Dough handling; Nutrition; Mediterranian; Dry fruits</td>
<td>East Asian; Salad; Cheese (hot); Fresh fruits; Curry; Creamy spreads; Mexican; Dry fruits; Grains; Jelly</td>
</tr>
<tr>
<td align="right">24</td>
<td>baked-goods dessert cobbler; baked-goods dessert pie; baked-goods dessert tart; dessert ; side-dishes pudding </td>
<td>Cakes; Egg &amp; dairy; Cookies; Dry fruits; Fruit tarts; Jelly; Chocolate; Dough handling; Breads; Sweet dessert; Fresh fruits; Dough (bread)</td>
<td>Meat; Poultry; Slow cooking; Curry; Creamy spreads; Mediterranian; Grains; Soup stock; Salad</td>
</tr>
<tr>
<td align="right">25</td>
<td>baked-goods pastry; ethnic europe austrian; holiday christmas </td>
<td>Dough (bread); Cookies; Cakes; Egg &amp; dairy; Chocolate; Breads; Dough handling; Asian spices; Dry fruits</td>
<td>Slow cooking; Mediterranian; Poultry; Mexican; Pasta; Soup stock; Curry; Salad; Microwave; Grilling</td>
</tr>
<tr>
<td align="right">26</td>
<td>holiday ; holiday halloween; misc camping; misc copycat; misc kid; misc mix; munchies snack </td>
<td>Dough (bread); Sweet dessert; Cookies; Chocolate; Breads; Jelly; Dough handling; Meat; Grains; Egg &amp; dairy; Legumes; Salad; Asian spices; Microwave; Mexican; Flatbreads; Creamy spreads</td>
<td>Frying; Cakes; Seafood; Mexican; Slow cooking; Curry; Egg &amp; dairy; Grilling; Microwave</td>
</tr>
<tr>
<td align="right">27</td>
<td>baked-goods dessert cooky; baked-goods scone </td>
<td>Cookies; Breads; Chocolate; Dough (bread); Cakes; Fresh fruits; Nutrition; Dry fruits; Dough handling</td>
<td>Grains; Microwave; Preservation; Grilling; Mexican; Salad; Jelly; Pasta; Curry; Cakes</td>
</tr>
<tr>
<td align="right">28</td>
<td>baked-goods bread; baked-goods bun; baked-goods muffin; baked-goods roll; misc pet-food dog; special-diets gluten-free </td>
<td>Breads; Dough (bread); Cookies; Cakes; Nutrition; Flatbreads; Grains</td>
<td>Poultry; Soup stock; Mexican; Curry; Mediterranian; Frying; Grilling; Legumes; Meat; Pasta; East Asian; Slow cooking</td>
</tr>
<tr>
<td align="right">29</td>
<td>baked-goods dessert cake; baked-goods dessert cheesecake </td>
<td>Cakes; Chocolate; Breads; Jelly; Sweet dessert; Dough (bread); Fruit tarts; Fresh fruits; Dough handling; Dry fruits</td>
<td>Creamy spreads; Slow cooking; Salad; Seafood; Grains; Mediterranian; Curry; Meat; Breads; Flatbreads; Cookies</td>
</tr>
<tr>
<td align="right">30</td>
<td>dessert frozen-desserts; dessert mousse; dessert trifle </td>
<td>Chocolate; Egg &amp; dairy; Jelly; Fresh fruits; Cakes; Sweet dessert; Fruit tarts; Dough (bread); Dough handling</td>
<td>East Asian; Mediterranian; Poultry; Grilling; Cookies</td>
</tr>
<tr>
<td align="right">31</td>
<td>dessert candy; dessert chocolate; holiday easter </td>
<td>Chocolate; Dough (bread); Sweet dessert; Jelly; Cookies; Cakes; Egg &amp; dairy; Dough handling; Breads</td>
<td>Salad; Flatbreads; Mediterranian; Frying; Soup stock; Seafood; Poultry; Asian spices</td>
</tr>
<tr>
<td align="right">32</td>
<td>baked-goods dessert brownie; dessert frosting </td>
<td>Chocolate; Cakes; Breads; Jelly; Cookies; Egg &amp; dairy; Nutrition</td>
<td>Slow cooking; Microwave; Flatbreads; Dough handling; Salad; Jelly; Cookies; Poultry; Asian spices; Breads</td>
</tr>
<tr>
<td align="right">33</td>
<td>misc canning </td>
<td>Preservation; Jelly; Dough (bread); Dry fruits; Legumes</td>
<td> </td>
</tr>
<tr>
<td align="right">34</td>
<td>side-dishes jam; side-dishes preserve </td>
<td>Jelly; Preservation; Fresh fruits; Asian spices; Dry fruits; Sweet dessert; Fruit tarts; Dough (bread); Blender</td>
<td>Seafood; Creamy spreads; Meat; Chocolate; Poultry; Soup stock; Egg &amp; dairy; Cheese (hot)</td>
</tr>
<tr>
<td align="right">35</td>
<td>dessert frozen-yogurt; dessert sherbet; dessert sorbet </td>
<td>Fresh fruits; Jelly; Blender; Fruit tarts; Chocolate; Cakes; Nutrition; Sweet dessert; Dry fruits</td>
<td>Asian spices; Microwave; East Asian; Salad; Cookies; Mexican; Grilling</td>
</tr>
<tr>
<td align="right">36</td>
<td>Fruits, grains &amp; veg: fruit; side-dishes beverage </td>
<td>Fresh fruits; Jelly; Dry fruits; Fruit tarts; Asian spices; Chocolate; Nutrition; Dough (bread); Dough handling; Egg &amp; dairy; Preservation</td>
<td>Grilling; Mediterranian; Creamy spreads; Cookies; Frying; Poultry; Curry; Dough handling; Sweet dessert; Seafood</td>
</tr>
</tbody></table>
</div>

Now that looks so much better!

For example, for cluster 2, which appears to be generically Asian, we could say that these are the traits that define "Asian" food:
<div style="height: 200px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Topic</th>
<th title="Field #2">Words</th>
</tr></thead>
<tbody><tr>
<td>East Asian</td>
<td>soy, sesam(e), sauc(e), chines(e), fr(y), oil, tofu, wok, tablespoon, ginger, sprout, chestnut, cornstarch, peanut, scallion</td>
</tr>
<tr>
<td> Seafood</td>
<td>fish, crab, seafood, fillet, shallot, scallop, lime, thai, mussel, lobster, squid, lemon, juic(e), francisco, san</td>
</tr>
<tr>
<td> Dough (bread)</td>
<td>dough, yeast, roll, rise, knead, warm, let, make, water, place, ball, hand, doubl(e), flour, work</td>
</tr>
<tr>
<td> Grilling</td>
<td>eggplant, grill, slice, roast, towel, paper, bacon, broil, place, cut, skewer, platter, broiler, avocado, skin</td>
</tr>
<tr>
<td> Meat</td>
<td>meat, beef, sauc(e), teaspoon, pork, ground, worchestershir(e), meatbal(l), tablespoon, pepper, patt(y), marinad(e), garlic, barbecu€</td>
</tr>
<tr>
<td> Frying</td>
<td>mushroom, onion, pepper, saut, heat, add, skillet, oil, minut(e), cook, tomato, rice, chop, garlic, stir</td>
</tr>
<tr>
<td> Salad</td>
<td>salad, dress(ing), vinegar, cucumb(er), mustard, lettuc(e), pepper, tablespoon, green, red, mayonnais(e), toss, dijon, ingredi(ent), prepar(e)</td>
</tr>
<tr>
<td> Soup stock</td>
<td>soup, stock, bay, simmer, carrot, broth, leek, leaf, add, celer(y), onion, stew, pot, boil, bring</td>
</tr>
<tr>
<td> Asian spices</td>
<td>spice, ginger, ground, allspic(e), cinnamon, clove, teaspoon, ml, raisin, chutney, seed, currant, cardamom, nutmeg, mace, store</td>
</tr>
<tr>
<td> Dough handling</td>
<td>cut, pastr(y), squash, foil, strip, inch, edg(e), knife, place, wrap, end, half, trim, piec(e), aluminum</td>
</tr>
<tr>
<td> Poultry</td>
<td>chicken, turkey, breast, poultr(y), stuf(fing), broth, casserol, skinless, bone, celer(y), sage, boneless, cook, mushroom, bird</td>
</tr>
<tr>
<td> Grains</td>
<td>rice, cup, lentil, stir, heat, prepar(e), cook, water, barley, boil, grain, saucepan, method, measur(e), add</td>
</tr>
<tr>
<td> Pasta</td>
<td>shrimp, noodl(e), ricotta, lasagna, spachetti, sauc(e), devein, pasta, mozzarella, lasagn(a), min, cook, chees(e), ounce, cottag(e)</td>
</tr>
</tbody></table>
</div>

and these are the traits that can be used to differentiate between different kinds of "Asian" food:
<div style="height: 200px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Topic</th>
<th title="Field #2">Words</th>
</tr></thead>
<tbody><tr>
<td>Flatbreads</td>
<td>pizza, bread, chees(e), bake, oven, crumb, phyllo, slice, sprinkl(e), crepe, preheat, mozzarella, sheet, parmesan, artichok(e)</td>
</tr>
<tr>
<td> Mediterranian</td>
<td>oliv(e), pasta, basil, oil, garlic, tomato, fresh, italian, parsley, pepper, herb, parmesan, pine, virgin, chop</td>
</tr>
<tr>
<td> Fruit tarts</td>
<td>pie, orang(e), pineappl(e), banana, crust, peach, cherr(y), shell, fruit, juic(e), whip, unbak(ed), raspberr(y), blueberr(y), dessert</td>
</tr>
<tr>
<td> Cheese (hot)</td>
<td>chees(e), cheddar, macaroni, broccoli, quich(e), shred, grate, bisquick, parmesan, cup, prepar(e), swiss, egg, zucchini, bake</td>
</tr>
<tr>
<td> Cakes</td>
<td>cake, sugar, cream, beat, vanilla, egg, pan, cup, cheesecak(e), bake, teaspoon, butter, prepar(e), cool, flour</td>
</tr>
<tr>
<td> Fresh fruits</td>
<td>strawberr(y), yogurt, mint, juic(e), lemon, berr(y), orang(e), honey, lime, fruit, fresh, plain, mango, zest, raspberr(y)</td>
</tr>
<tr>
<td> Sweet dessert</td>
<td>pumpkin, peanut, cracker, pud(ding), mix, graham, milk, cup, condens(e), prepar(e), sweeten, jello, cereal, whip, key</td>
</tr>
<tr>
<td> Preservation</td>
<td>jar, pint, ft, canner, canning, pressu(re), process, pickl(e), tabl(e), lid, recommend, altitud(e), quart, pound, headspac(e)</td>
</tr>
<tr>
<td> Chocolate</td>
<td>chocol(ate), cocoa, chip, cake, cand(y), vanilla, melt, browni(es), cream, frost, coffe(e), semisweet, sugar, marshmallow, cool</td>
</tr>
<tr>
<td> Creamy spreads</td>
<td>dip, chees(e), cream, sour, sandwich, tuna, appet(ite), mayonnais(e), salmon, spread, ounce, pita, dill, cracker, hor</td>
</tr>
<tr>
<td> Slow cooking</td>
<td>crockpot, crock, pot, oyster, cabbag(e), steak, cooker, slow, clam, hour, beef, cajun, meat, sausag(e), cook</td>
</tr>
<tr>
<td> Cookies</td>
<td>cooki(es), bake, flour, sheet, sugar, dough, teaspoon, butter, soda, roll, shorten, egg, brown, purpos(e), oven</td>
</tr>
</tbody></table>
</div>

At first glance it seems odd that "grains" are not higher up on the list of inter-cluster differentiating topics, since rice is so important to Asians. And "pasta" is there right next to grains. But if you look more closely, the "grains" topic also includes other kinds of grains, and you do have rice dishes around the world - so just rice alone isn't that good a differentiating factor for Asian food. And the "pasta" topic also includes the generic "noodle", as well as "prawn" and "sauce", which are pretty important for Asian food. More importantly, the use of sauces in general, sesame, soy, and frying are really central to a lot of Asian foods.

On the other hand, things like various breads, garlic, olive, fruits, desserts, baking, and cheeses are part of Asian cooking, but not all that common. So it would make sense that these are possible ways to differentiate between different types of Asian cooking.

I can't comment on all the various European regional cuisines, but in general, this set of intra-cluster differentiating topics seem to be reasonably good differentiators.

So that's it, I guess. If you want to get some ideas of how recipes from different parts of the world differ, you can take this approach, and identify key ingredients and techniques used. You could also apply it on an individual recipe basis, but you'll need to find some other way to visualise it clearly.

I'm more or less done with this project, but here are some avenues I'd like to explore in future:
<ul>
  <li>Testing this on other databases - unfortunately, I'll need to find some way to label the recipes from other databases!</li>
  <li>Machine Learning: I've tried running artificial neural networks on the tf-idf values of the raw corpus (54000+ recipes) randomly split into training and testing sets, but I haven't been able to get above 75% accuracy on the test set. I have not tried doing this with the 35 topic weights instead of using thousands of word counts for each of the 54000+ recipes.</li>
  <li>Web functionality - would be cool if I could upload this on a server so that people can type in a recipe URL and it'll extract the text from that URL and try to guess what type of recipe it is.</li>
