---
layout: post
title: "Analysing similarities between recipes with topic modeling"
date: 2019-11-12
---
<i>The Python code used for this post can be found here: <a>https://github.com/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191112/recipe-analysis.py</a></i>

I'm a Singaporean, which means that appreciating the vast diversity of food is a way of life for me. Growing up in a melting pot of cultures and travelling overseas to savour the taste of various local delicacies does that for you.

Sometimes you hear people say things like "This doesn't taste authentic!" 

Well, then what makes something authentic? As an avid amateur chef, my best guess is that (other than sociocultural factors such as who's cooking it and where you're eating it) it's in the ingredients and cooking methods. For example, I don't expect to see Chinese recipes call for letting dough rise before baking, and neither do I expect to see thinly sliced raw fish in American recipes. But I'd like to quantify this.

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

So I went on to make use of topic modeling using the Gensim library. <a href="https://not-even-wong.github.io/2019/11/10/layman-intro-topic-modeling.html">I wrote a very quick layman primer to it here</a>.

After some experimenting to decide the number of topics, I settled on 35 topics: too few resulted in topics not being specific enough; too many resulted in too much overlap between topics. I used pyLDAvis to visualise these topics.

<a href="https://nbviewer.jupyter.org/github/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191112/pyLDAvis%20visual%20for%20upload.ipynb">View online</a>

<a href="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/lda35.html">Download to desktop</a>

I set λ = 0.6 for a good balance between term frequency and uniqueness, and used that to assign labels to each topic:

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

Now that I have a list of topics, I can use these to make sense of clustering. As with my earlier attempt, I merged all documents within each category of recipes into a single long document. I then used gensim's LDA model to assign weights for each topic for each document. I then used these to perform heirarchical clustering to obtain this result:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191112/heirarchy%20of%20recipes%20with%20gensim%20topics.png">

Finally, let's see what topics contribute to each cluster! Based on the results of the heirarchical clustering, I'll define the following clusters:


