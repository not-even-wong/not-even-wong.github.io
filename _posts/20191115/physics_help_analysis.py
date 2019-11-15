import pandas as pd
import numpy as np

import gensim
from gensim import corpora
import pyLDAvis.gensim

import re# Remove punctuation
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.feature_extraction import text
from textblob import TextBlob

import pickle
import webbrowser

"""import data"""
modeling_data=pd.read_csv('corpus_merged.csv', index_col=0)['Text']
corpuslength=len(modeling_data)
modeling_data=modeling_data[3:corpuslength]
modeling_data=modeling_data.reset_index()['Text']

"""clean text"""

#remove numerals
corpus_raw = modeling_data.str.replace('\d+','')

#remove webpages
"""add in something that removes between url tags?"""
webpage_suffixes=np.array(['.com','.net','.gov','.org','.edu','.html','.htm'])
for j in webpage_suffixes:
    for i in range(0,len(corpus_raw)):
        end_index=corpus_raw[i].find(j)
        while end_index!=-1:
            start_index=corpus_raw[i][:end_index].rfind(' ')
            print('webpage removed: '+corpus_raw[i][start_index:end_index+4])
            corpus_raw[i]="".join((corpus_raw[i][:start_index]," ",corpus_raw[i][end_index+4:]))
            end_index=corpus_raw[i].find(j)
            
webpage_prefixes=np.array(['http','www'])
for j in webpage_prefixes:
    for i in range(0,len(corpus_raw)):
        start_index=corpus_raw[i].find(j)
        while start_index!=-1:
            end_index=start_index+corpus_raw[i][start_index:].find(' ')
            print('webpage removed: '+corpus_raw[i][start_index:end_index])
            corpus_raw[i]="".join((corpus_raw[i][:start_index]," ",corpus_raw[i][end_index:]))
            start_index=corpus_raw[i].find(j)

#remove punctuation and set all lowercase
corpus_raw = corpus_raw.map(lambda x: re.sub('[,\.!?*()-:]', ' ', x))
corpus_raw = corpus_raw.map(lambda x: x.lower())

#remove bad words
corpus_raw = corpus_raw.str.replace('/',' ') #separate words with /, e.g. hot/cold
corpus_raw = corpus_raw.str.replace('\\',' ') #separate words with /, e.g. hot/cold
corpus_raw = corpus_raw.str.replace('-','') #join words with -, e.g. non-stick
corpus_raw = corpus_raw.str.replace('\'','') #remove '
corpus_raw = corpus_raw.str.replace('el nino','') 

my_stop_words=text.ENGLISH_STOP_WORDS.union(['tex','...','…','protein','olderdan','gold','ga','pf','hi','welcom','promis','©','english','fg','rl','bhat','daniel','learningphys','gh','did','itex','rm', 'acid', 'song','wa','•','niño','el','biologist','ocean','extinct','question', 'forum', 'pinterest', 'topic', 'email', 'whatsapp', 'twitter', 'facebook', 'share', 'jisbon', 'reddit', 'click', 'just', 'said', 'know', 'make', 'reply', 'repli', 'thank', 'refer', 'want', 'someth', 'advisor', 'edit', 'experi', 'ask', 'book', 'good', 'explain', 'read', 'example', 'year', 'teacher', 'basic', 'come', 'helper', 'member', 'email', 'jisbon', 'kaushik', 'mentor', 'because', 'ok', 'player', 'helper', 'messag', 'need', 'answer', 'rss', 'attempt', 'award', 'learn', 'emeritu', 'kuruman', 'jeff', 'moara', 'compassion', 'civil', 'grammar', 'solo', 'discipline', 'privacy', 'recognit', 'mainstream', 'patienc', 'edit', 'got', 'need', 'English', 'service', 'spell', 'doe', 'gneill', 'intern', 'because', 'got', 'Kaushik', 'think', 'child', 'provid', 'Kaushik', 'English', 'privaci', 'songbird', 'neuron', 'nino', 'climat', 'servic', 'debat', 'disciplin', 'aisha', 'barbel', 'pleas', 'post', 'tarzan', 'pilot', 'skateboard', 'freak', 'cupid', 'rollercoast', 'helicopt', 'conveyor', 'ehild', 'berkeman', 'ideasrul', 'wiki', 'author', 'insight', 'haruspex', 'png', 'orodruin', 'jahnavi', 'mathbf', 'upload', 'pushoam', 'chestermil', 'sorri', 'mathrm', 'songoku', 'quit', 'rude', 'hallsofivi', 'dont', 'hint', 'arildno', 'whozum', 'dearli', 'marlon', 'urbanxrisi', 'pyrrhu', 'mattson', 'dexterciobi', 'cookiemonst', 'gokul', 'phanthomjay', 'basebal', 'tenni', 'footbal', 'golf', 'text', 'mistak', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'flyingpig', 'becaus', 'understand', 'thing', 'andrew', 'mason', 'phanthomjay', 'appreci', 'hootenanni', 'boat', '\'m', '\'s', 'n\'t'])
my_stop_words=[x for x in my_stop_words]#justify: acid, song, ocean, etc. only appear in very few questions

pickle.dump(corpus_raw, open('corpus_raw191114.pkl', 'wb'))
corpus_raw=pickle.load(open('corpus_raw191114.pkl', 'rb'))

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens if token.stem() not in my_stop_words]
    return words

#tokenize, stem, stopword removal
corpus_raw_token = [textblob_tokenizer(doc) for doc in corpus_raw]

"""perform LDA"""
dictname='dictionary'+str(corpuslength)+'.gensim'
dictionary = corpora.Dictionary(corpus_raw_token)
dictionary.filter_extremes(no_above=0.9,no_below=round(0.01*len(corpus_raw_token)))
corpus = [dictionary.doc2bow(text) for text in corpus_raw_token]

pickle.dump(corpus_raw_token, open('corpus_raw_token191114.pkl', 'wb'))
pickle.dump(corpus, open('corpus191114.pkl', 'wb'))
pickle.dump(dictionary,open('dictionary191114.pkl','wb'))
dictionary.save(dictname)

corpus_raw_token=pickle.load(open('corpus_raw_token191114.pkl', 'rb'))
corpus=pickle.load(open('corpus191114.pkl', 'rb'))
dictionary=pickle.load(open('dictionary191114.pkl','rb'))

#dictionary = gensim.corpora.Dictionary.load('dictionary104538.gensim')
#corpus = pickle.load(open('corpus104538.pkl', 'rb'))

#NUM_TOPICS = 15

def displaytopics(NUM_TOPICS,dictionary,corpus):
    modelname = 'model'+str(NUM_TOPICS)
    displayname = 'lda'+str(NUM_TOPICS)
    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, workers = None)
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, displayname+'.html')
    webbrowser.open(displayname+'.html',new=2)
    pickle.dump(ldamodel,open(modelname+'.pkl','wb'))
    pickle.dump(lda_display,open(displayname+'.pkl','wb'))

displaytopics(65,dictionary,corpus)

"""get weights"""
raw_data=pd.read_csv('corpus_merged.csv', index_col=0)
corpuslength=len(raw_data)
raw_data=raw_data[3:corpuslength]
raw_data=raw_data.reset_index()
