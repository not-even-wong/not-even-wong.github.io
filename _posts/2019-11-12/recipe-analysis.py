import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction import text
from textblob import TextBlob

import gensim
from gensim import corpora
import pyLDAvis.gensim

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

import pickle
import webbrowser

"""import data"""
raw_data = pd.read_csv('recipe_raw_data_categorised.csv')

"""format data:"""

#merge titles
for i in range(len(raw_data)):
    Cat = str(raw_data.iat[i,2])+str(' ')+str(raw_data.iat[i,3])+str(' ')+str(raw_data.iat[i,4])+str(' ')+str(raw_data.iat[i,5])
    Cat = Cat.replace('nan','')
    Cat = Cat.replace('fgv','Fruits, grains & veg:')
    raw_data.at[i,'Cat']=Cat

for i in range(1,5):
    name='cat'+str(i)    
    raw_data=raw_data.drop(name,axis=1)
raw_data=raw_data.drop('0',axis=1)

#remove categories with too few samples

Cat_list=pd.DataFrame(raw_data['Cat'].value_counts(ascending=True))
Cat_list.to_csv(r'cat_list_for_merging.csv')

#remove categories with too few entries
Removal_list = Cat_list.iloc[:20]
Removal_list = pd.DataFrame(Removal_list.index)

print('remove the following categories:')
print(Removal_list)

for i in range(0,len(Removal_list)):
    raw_data=raw_data[raw_data.Cat != Removal_list.iat[i,0]]    

"""create and clean corpus"""
raw_data=raw_data.reset_index()
corpus = raw_data['1'].str.replace('\d+',' ')

#remove webpages
webpage_suffixes=np.array(['.com','.net','.gov','.org','.co','.edu','.html','.htm'])
for j in webpage_suffixes:
    for i in range(0,len(corpus)):
        end_index=corpus[i].find(j)
        while end_index!=-1:
            start_index=corpus[i][:end_index].rfind(' ')
            print('webpage removed: '+corpus[i][start_index:end_index+4])
            corpus[i]="".join((corpus[i][:start_index]," ",corpus[i][end_index+4:]))
            end_index=corpus[i].find(j)
            
webpage_prefixes=np.array(['http','www'])
for j in webpage_prefixes:
    for i in range(0,len(corpus)):
        start_index=corpus[i].find(j)
        while start_index!=-1:
            end_index=start_index+corpus[i][start_index:].find(' ')
            print('webpage removed: '+corpus[i][start_index:end_index])
            corpus[i]="".join((corpus[i][:start_index]," ",corpus[i][end_index:]))
            start_index=corpus[i].find(j)

#remove punctuation and set all lowercase
corpus = corpus.map(lambda x: re.sub('[,\.!?*()-:]', ' ', x))
corpus = corpus.map(lambda x: x.lower())

#remove problem words
corpus = corpus.str.replace('/',' ') #separate words with /, e.g. hot/cold
corpus = corpus.str.replace('\\',' ') #separate words with /, e.g. hot/cold
corpus = corpus.str.replace('-','') #join words with -, e.g. non-stick
corpus = corpus.str.replace('medium','med') #standardise measurement word
corpus = corpus.str.replace('upside down','upsidedown') #standardise measurement word
corpus = corpus.str.replace('chick pea','chickpea') #standardise term
corpus = corpus.str.replace('children','child') #standardise term
corpus = corpus.str.replace('knives','knife') #standardise term
corpus = corpus.str.replace('chili','chilli') #standardise term
corpus = corpus.str.replace('crab apple','crabapple') #standardise term
corpus = corpus.str.replace('crabmeat','crab') #standardise term
corpus = corpus.str.replace('minutes.\x14','minutes') #remove stubborn typo
corpus = corpus.str.replace('mealmaster',' ') #remove common irrelevant word
corpus = corpus.str.replace('women','woman') #standardise term
corpus = corpus.str.replace('worchestershire','worcestershire') #standardise term
corpus = corpus.str.replace('yoghurt','yogurt') #standardise term
corpus = corpus.str.replace(' tsp',' teaspoon') #standardise abbreviation
corpus = corpus.str.replace(' tbsp',' tablespoon') #standardise abbreviation
corpus = corpus.str.replace(' ts',' teaspoon') #standardise abbreviation
corpus = corpus.str.replace(' tb',' tablespoon') #standardise abbreviation
corpus = corpus.str.replace(' oz',' ounce') #standardise abbreviation
corpus = corpus.str.replace(' pt',' pint') #standardise abbreviation
corpus = corpus.str.replace(' lb',' pound') #standardise abbreviation
corpus = corpus.str.replace(' qt',' quart') #standardise abbreviation
corpus = corpus.str.replace('milligram','mg') #standardise abbreviation
corpus = corpus.str.replace(' gram',' g') #standardise abbreviation

#problem: C can be cup or celsius. Just remove wholesale, since it won't provide useful info

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens if token.stem() not in my_stop_words]
    return words

my_stop_words=text.ENGLISH_STOP_WORDS.union(["steiger","o'brion","like","good","want","day","t","ea","c","note","fabfood","site","xbrg","mcfarland","b","v","geffen","bb","vghc","waldin","titl","tm","mc","karen","mintzia","wa","gm","master","thi","use","s","”","“","’","\'",'\"',"mastercook","exported","mmmmm","",'copyright','photograph','recipelu', 'repost', 'reprint', 'request', 's.c', 's.smith', 'typo','usenet', 'abbott', 'acd.net', 'airmail.net', 'archer-daniels-midland', 'atbnb', 'best.com', 'billspa', 'brigitt', 'bwvbb', 'chpd', 'cjhartlin', 'cyberealm', 'deeann', 'earthlink.net', 'hcpmc', 'http', 'ima.infomail.com', 'illustr', 'icanect.net', 'idsonline.com', 'ihnp', 'isbn', 'ix.netcom.com', 'jphelp', 'juno.com', 'masterchef', 'mc-recip', 'mc_buster', 'mcrecip', 'micro-cook', 'mmconv', 'mmformat', 'msn.com', "n't",'netrax.net', 'archerdanielsmidland', 'mcrecip', 'microcook', 'npxrb', 'pjxga', 'prodigi', 'prodigy.com', 'r.d', 'rec.food.cook', 'rec.food.recip', 'reggie.com', 'rfvc', 'salata.com', 'shell.portal.com', 'sojourn.com', 'the.steig', 'tx', 'txfta', 'vdrja', 'vghca', 'vidalia', 'websit', 'webster', 'wizard.ucr.edu', 'wrv', 'wv', 'ww', 'www.hersheys.com', 'x', 'x-inch', 'xbrga', 'xx', 'xx-inch', 'y', 'yy', 'yyyyyyyyyyyyyyyyyi', 'z','mealmaster','mealmast','trademark'])
my_stop_words=[x for x in my_stop_words]

#tokenize, stem and remove words:
corpus_raw = [textblob_tokenizer(doc) for doc in corpus]

#save backup file
pickle.dump(corpus_raw, open('corpus_raw.pkl','wb'))

#corpus_raw = pickle.load(open('corpus_raw.pkl', 'rb'))

"""perform LDA"""

#create dictionary and bag-of-words representation
dictname='dictionary.gensim'
dictionary = corpora.Dictionary(corpus_raw)
dictionary.filter_extremes(no_above=0.9,no_below=round(0.001*len(corpus_raw)))
corpus_bow = [dictionary.doc2bow(text) for text in corpus_raw]

#save backup file:
pickle.dump(corpus_bow, open('corpus_bow.pkl', 'wb'))
dictionary.save(dictname)

#dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
#corpus_bow = pickle.load(open('corpus_bow.pkl', 'rb'))

#create LDA model
NUM_TOPICS = 35
modelname = 'model'+str(NUM_TOPICS)
displayname = 'lda'+str(NUM_TOPICS)+'.html'

ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus_bow, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, workers = None)
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus_bow, dictionary, sort_topics=False, n_jobs=-1)#, R=15)
pyLDAvis.save_html(lda_display, displayname)
webbrowser.open(displayname,new=2)

#save backup
pickle.dump(ldamodel, open('ldamodel.pkl', 'wb'))

#save topics and terms
topics=pd.DataFrame(ldamodel.print_topics(num_topics=NUM_TOPICS, num_words=30)).drop(columns=0)

for i in range(len(topics)):
    text=topics.iat[i,0]
    topics.iat[i,0]=text.replace('*',' ') #clean for easier reading

topics.to_csv('gensim_topics.csv',index=False,header=False)


"""apply LDA model to each consolidated document for each category"""

#merge data
combined_text=pd.DataFrame(raw_data['Cat'].unique().tolist())

corpusdf=combined_text.copy(deep=True)

for i in range(0,len(combined_text)):
    Cat=combined_text.iat[i,0]
    corpusdf.iat[i,0]=raw_data[raw_data['Cat'].str.match(Cat)]['1'].str.cat(sep=' ')

#clean data
corpusdf=corpusdf[0]
corpusdf = corpusdf.str.replace('/',' ') #separate words with /, e.g. hot/cold
corpusdf = corpusdf.str.replace('\\',' ') #separate words with /, e.g. hot/cold
corpusdf = corpusdf.str.replace('-','') #join words with -, e.g. non-stick
corpusdf = corpusdf.str.replace('medium','med') #standardise measurement word
corpusdf = corpusdf.str.replace('upside down','upsidedown') #standardise measurement word
corpusdf = corpusdf.str.replace('chick pea','chickpea')
corpusdf = corpusdf.str.replace('children','child')
corpusdf = corpusdf.str.replace('knives','knife')
corpusdf = corpusdf.str.replace('chili','chilli')
corpusdf = corpusdf.str.replace('crab apple','crabapple')
corpusdf = corpusdf.str.replace('crabmeat','crab')
corpusdf = corpusdf.str.replace('minutes.\x14','minutes')
corpusdf = corpusdf.str.replace('mealmaster',' ')
corpusdf = corpusdf.str.replace('women','woman')
corpusdf = corpusdf.str.replace('worchestershire','worcestershire')
corpusdf = corpusdf.str.replace('yoghurt','yogurt')
corpusdf = corpusdf.str.replace(' tsp',' teaspoon')
corpusdf = corpusdf.str.replace(' tbsp',' tablespoon')
corpusdf = corpusdf.str.replace(' ts',' teaspoon')
corpusdf = corpusdf.str.replace(' tb',' tablespoon')
corpusdf = corpusdf.str.replace(' oz',' ounce')
corpusdf = corpusdf.str.replace(' pt',' pint')
corpusdf = corpusdf.str.replace(' lb',' pound')
corpusdf = corpusdf.str.replace(' qt',' quart')
corpusdf = corpusdf.str.replace('milligram','mg')
corpusdf = corpusdf.str.replace(' gram',' g')

#tokenize and stem
corpusdf_raw = [textblob_tokenizer(doc) for doc in corpusdf]

#save backup
pickle.dump(corpusdf_raw, open('corpusdf_raw.pkl','wb'))

#create bag of words
corpusdf_bow=[dictionary.doc2bow(text) for text in corpusdf_raw]

#determine topic weights for each category
cat_scores=pd.DataFrame(pd.DataFrame(ldamodel.get_document_topics(corpusdf_bow[0], minimum_probability=0.0))[1]).transpose()
for i in range(1,len(corpusdf_bow)):
    new_row=pd.DataFrame(pd.DataFrame(ldamodel.get_document_topics(corpusdf_bow[i], minimum_probability=0.0))[1]).transpose()
    cat_scores=cat_scores.append(new_row,ignore_index=True)

#export document showing weights for each category
cat_score_label=combined_text.rename(columns={0:'cat'}).join(cat_scores)
cat_score_label.to_csv(r'cats_scored.csv')

"""create heirarchical clustering"""

#define labels for axis
categories = combined_text.to_numpy(dtype=str)
categories = pd.DataFrame(categories).to_numpy()

#calculate heirarchical relationships
heirarchy = linkage(cat_scores,'ward')

#generate graph
plt.figure(figsize=(50, 4))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('type of recipe')
plt.ylabel('distance')
dendrogram(
    heirarchy,
    color_threshold=0.7, #cutoff for 'clusters'
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=12.,  # font size for the x axis labels
    labels=np.ravel(categories) #labels each leaf
)
plt.show()
