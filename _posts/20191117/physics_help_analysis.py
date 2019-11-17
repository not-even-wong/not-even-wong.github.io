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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D      
from matplotlib import cm

from datetime import date

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

pickle.dump(corpus_raw, open('corpus_raw191114.pkl', 'wb'))
corpus_raw=pickle.load(open('corpus_raw191114.pkl', 'rb'))

my_stop_words=text.ENGLISH_STOP_WORDS.union(['tex','...','…','protein','olderdan','gold','ga','pf','hi','welcom','promis','©','english','fg','rl','bhat','daniel','learningphys','gh','did','itex','rm', 'acid', 'song','wa','•','niño','el','biologist','ocean','extinct','question', 'forum', 'pinterest', 'topic', 'email', 'whatsapp', 'twitter', 'facebook', 'share', 'jisbon', 'reddit', 'click', 'just', 'said', 'know', 'make', 'reply', 'repli', 'thank', 'refer', 'want', 'someth', 'advisor', 'edit', 'experi', 'ask', 'book', 'good', 'explain', 'read', 'example', 'year', 'teacher', 'basic', 'come', 'helper', 'member', 'email', 'jisbon', 'kaushik', 'mentor', 'because', 'ok', 'player', 'helper', 'messag', 'need', 'answer', 'rss', 'attempt', 'award', 'learn', 'emeritu', 'kuruman', 'jeff', 'moara', 'compassion', 'civil', 'grammar', 'solo', 'discipline', 'privacy', 'recognit', 'mainstream', 'patienc', 'edit', 'got', 'need', 'English', 'service', 'spell', 'doe', 'gneill', 'intern', 'because', 'got', 'Kaushik', 'think', 'child', 'provid', 'Kaushik', 'English', 'privaci', 'songbird', 'neuron', 'nino', 'climat', 'servic', 'debat', 'disciplin', 'aisha', 'barbel', 'pleas', 'post', 'tarzan', 'pilot', 'skateboard', 'freak', 'cupid', 'rollercoast', 'helicopt', 'conveyor', 'ehild', 'berkeman', 'ideasrul', 'wiki', 'author', 'insight', 'haruspex', 'png', 'orodruin', 'jahnavi', 'mathbf', 'upload', 'pushoam', 'chestermil', 'sorri', 'mathrm', 'songoku', 'quit', 'rude', 'hallsofivi', 'dont', 'hint', 'arildno', 'whozum', 'dearli', 'marlon', 'urbanxrisi', 'pyrrhu', 'mattson', 'dexterciobi', 'cookiemonst', 'gokul', 'phanthomjay', 'basebal', 'tenni', 'footbal', 'golf', 'text', 'mistak', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'flyingpig', 'becaus', 'understand', 'thing', 'andrew', 'mason', 'phanthomjay', 'appreci', 'hootenanni', 'boat', '\'m', '\'s', 'n\'t'])
my_stop_words=[x for x in my_stop_words]
#Note: acid, song, ocean, etc. are prevailent only due to advertisments during the time of scraping: physicsforums recommended several posts

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

corpus_raw_token=pickle.load(open('corpus_raw_token191114.pkl', 'rb'))
corpus=pickle.load(open('corpus191114.pkl', 'rb'))
dictionary=pickle.load(open('dictionary191114.pkl','rb'))

def displaytopics(NUM_TOPICS,dictionary,corpus):
    modelname = 'model'+str(NUM_TOPICS)
    displayname = 'lda'+str(NUM_TOPICS)
    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, workers = None)
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, displayname+'.html')
    webbrowser.open(displayname+'.html',new=2)
    pickle.dump(ldamodel,open(modelname+'.pkl','wb'))
    pickle.dump(lda_display,open(displayname+'.pkl','wb'))

displaytopics(60,dictionary,corpus)

"""get weights"""
raw_data=pd.read_csv('corpus_merged.csv', index_col=0)
corpuslength=len(raw_data)
raw_data=raw_data[3:corpuslength]
raw_data=raw_data.reset_index()

ldamodel=pickle.load(open('model60.pkl','rb'))
thread_scores=pd.DataFrame(pd.DataFrame(ldamodel.get_document_topics(corpus[0], minimum_probability=0.0))[1]).transpose()
for i in range(1,len(corpus)):
    new_row=pd.DataFrame(pd.DataFrame(ldamodel.get_document_topics(corpus[i], minimum_probability=0.0))[1]).transpose()
    thread_scores=thread_scores.append(new_row,ignore_index=True)

ldamodel=pickle.load(open('model65.pkl','rb'))
thread_scores65=pd.DataFrame(pd.DataFrame(ldamodel.get_document_topics(corpus[0], minimum_probability=0.0))[1]).transpose()
for i in range(1,len(corpus)):
    new_row=pd.DataFrame(pd.DataFrame(ldamodel.get_document_topics(corpus[i], minimum_probability=0.0))[1]).transpose()
    thread_scores65=thread_scores65.append(new_row,ignore_index=True)

thread_scores.columns=np.arange(1,61)
thread_scores[61]=thread_scores65[13] #appending "rotational motion" from another model

thread_scores.to_csv(r'thread_scores.csv',index=False,header=False)

topic_list=pd.read_csv(r'list of topic names.csv',header=None)
thread_scores=pd.read_csv(r'thread_scores.csv',header=None)
thread_scores.columns=np.arange(1,62)
all_data=raw_data.join(thread_scores)

all_data.to_csv(r'all_weights.csv',index=False)

"""Analysis of weights performed below"""

all_data=pd.read_csv(r'all_weights.csv', index_col=0).reset_index()
all_data_backup=all_data.copy(deep=True)
all_data=all_data_backup.copy(deep=True)
topic_list=pd.read_csv(r'list of topic names.csv',header=None)

#the following topics are not relevant: 18, 34, 39, 54 (advertisments)
#the following topics are about math: 15, 22, 24, 27, 38, 41, 43, 45, 46, 47, 51, 52, 53
#the following topics are language use: 23 about school, 50 about clarification and questions, 55 MCQ choices, 56 file attachments

"""convert views from K to 1000s (e.g. 10k -> 10000)"""
pd.set_option('display.max_columns', 20)

for i in range(len(all_data)):
    try:
        if all_data.at[i,'Views'].find('K')!=-1:
            all_data.at[i,'Views']=int(all_data.at[i,'Views'][0:len(all_data.at[i,'Views'])-1])*1000
            #print('row '+str(i)+' to '+str(all_data.at[i,'Views']))
        else:
            all_data.at[i,'Views']=int(all_data.at[i,'Views'])
    except:
        print('error '+str(i))

all_data=all_data.drop(234,0).reset_index().drop(['index','level_0'],1)
print(all_data[['URL','Date','Views','Replies']][220:230])

for i in range(len(all_data)):
    all_data.at[i,'Replies']=float(all_data.at[i,'Replies'])

"""calculate number of days at time of scraping"""
#data was scraped 22 oct

for i in range(len(all_data)):
    all_data.at[i,'Year']=int(all_data.at[i,'Date'][0:4])
    all_data.at[i,'Month']=int(all_data.at[i,'Date'][5:7])
    all_data.at[i,'Day']=int(all_data.at[i,'Date'][8:10])
    all_data.at[i,'Year-Month']=all_data.at[i,'Date'][0:7]
    print(i)

def days_passed(row):
    return ((date(2019, 10, 22)-date(int(all_data.at[row,'Year']),int(all_data.at[row,'Month']),int(all_data.at[row,'Day']))).days)
def months_passed(row):
    years=2019-all_data.at[row,'Year']
    months=10-all_data.at[row,'Month']
    return(12*years+months)

for i in range(len(all_data)):
    all_data.at[i,'Days']=days_passed(i)
    all_data.at[i,'Months']=months_passed(i)
    print(str(i)+': '+str(all_data.at[i,'Days']))

#show views over time
plt.figure(figsize=(12,12))
plt.scatter(all_data['Days'],all_data['Views'],marker='2')
plt.xlabel('Age of thread in days')
plt.ylabel('Views')
plt.show()

#show replies over time
plt.figure(figsize=(12,12))
plt.scatter(all_data['Days'],all_data['Replies'],marker='2')
plt.xlabel('Age of thread in days')
plt.ylabel('Replies')
plt.show()

print(all_data.sort_values('Views',ascending=False)[['URL','Views']])
# outliers with exceptionally large number of views are:
# converting-converting-km-hr-to-m-s.200055
# tension-formula.329194
# need-egg-drop-project-ideas.186507
# find-spring-constant-of-spring-in-n-m-help.99506
# phase-difference-mutual-inductor.526105

"""identify relationships between topics"""
#get correlation matrix
topics_cov=np.corrcoef(all_data[np.arange(1,62).astype(np.str)].transpose())

plt.figure(figsize=(12,12))
plt.imshow(topics_cov,cmap='plasma')
plt.xticks(np.arange(61),np.ravel(topic_list),rotation='vertical')
plt.yticks(np.arange(61),np.ravel(topic_list))
plt.xlim(-0.5,60.5)
plt.ylim(-0.5,60.5)
plt.title('Correlation between topics')
plt.grid(True)
plt.show()

#the advertisments are skewing the data...
#the following topics are not relevant: 18, 34, 39, 54
#the following topics are language: 23 about school, 50 about clarification and questions, 55 MCQ choices, 56 file attachments

irrelevant=np.array([17,33,38,53,22,49,54,55])
cleaned_cov_data=all_data[np.arange(1,62).astype(np.str)].transpose().drop(irrelevant.astype(np.str),0)
cleaned_topic_list=topic_list.drop(irrelevant,0)

clean_topics_cov=np.corrcoef(cleaned_cov_data)

plt.figure(figsize=(12,12))
plt.imshow(clean_topics_cov,cmap='plasma')
plt.xticks(np.arange(53),np.ravel(cleaned_topic_list),rotation='vertical')
plt.yticks(np.arange(53),np.ravel(cleaned_topic_list))
plt.xlim(-0.5,52.5)
plt.ylim(-0.5,52.5)
plt.title('Covariance of topics')
plt.grid(True)
plt.show()

correlation_list=pd.DataFrame(['Topic 1','Index 1','Topic 2','Index 2','Correlation']).transpose()
for i in range(len(clean_topics_cov)):
    for j in range(i+1,len(clean_topics_cov)):
        newline=pd.DataFrame([cleaned_topic_list.iat[i,0],i,cleaned_topic_list.iat[j,0],j,clean_topics_cov[i,j]]).transpose()
        correlation_list=correlation_list.append(newline)
correlation_list.columns=['Topic 1','Index 1','Topic 2','Index 2','Correlation']
correlation_list=correlation_list.reset_index().drop(0,0).drop('index',1)
correlation_list=correlation_list.sort_values('Correlation',ascending=False).reset_index().drop('index',1)
print(correlation_list)        

correlation_list.to_csv(r'correlation between topics.csv')

for i in range(5):
    print('r = '+str(correlation_list.iat[i,4]))
    plt.figure(figsize=(6,6))
    plt.scatter(all_data[str(correlation_list.iat[i,1])],all_data[str(correlation_list.iat[i,3])],marker='x')
    plt.xlabel(correlation_list.iat[i,0])
    plt.ylabel(correlation_list.iat[i,2])
    plt.show()
    
for i in range(5):
    print('r = '+str(correlation_list.iat[1377-i,4]))
    plt.figure(figsize=(6,6))
    plt.scatter(all_data[str(correlation_list.iat[1377-i,1])],all_data[str(correlation_list.iat[1377-i,3])],marker='x')
    plt.xlabel(correlation_list.iat[1377-i,0])
    plt.ylabel(correlation_list.iat[1377-i,2])
    plt.show()

#no strong relationships shown
    
dep_var=all_data[['Views','Replies']].astype('int64')
ind_var=cleaned_cov_data.transpose()

popularity_prediction=pd.DataFrame(np.corrcoef(dep_var.join(ind_var).transpose())).iloc[:,0:2].drop([0,1],0).reset_index().drop('index',1)
popularity_prediction.columns=['Views','Replies']

print(popularity_prediction.sort_values('Views'))
print(popularity_prediction.sort_values('Replies'))

popularity_prediction.to_csv('popularity_correlation.csv')

#it seems that views are very not correlated to topics
#replies are weakly correlated to topic 30, Gauss's law, and nothing else
#furthermore, apparently no correlation between views and replies

plt.figure(figsize=(12,12))
plt.scatter(all_data['Views'],all_data['Replies'],marker='2')
plt.xlabel('Views')
plt.ylabel('Replies')
plt.show()

plt.figure(figsize=(12,12))
plt.scatter(all_data['30'],all_data['Replies'],marker='2')
plt.xlabel('Weight of Topic 30: Gauss\'s Law')
plt.ylabel('Replies')
plt.show()

"""analyse trends of topics over time"""
"""aggregate information by year and month"""

chronology=pd.DataFrame(all_data['Year-Month'].unique(),columns=['Year-Month']).sort_values('Year-Month').reset_index()
chronology=chronology.drop('index',1)

for i in range(len(chronology)):
    sliced_data=all_data.loc[all_data['Year-Month']==chronology.iat[i,0]]
    for j in range(1,62):
        chronology.at[i,'Months']=sliced_data.reset_index().at[0,'Months']
        chronology.at[i,'n']=len(sliced_data[str(j)])
        chronology.at[i,j]=sliced_data[str(j)].mean()

print(chronology[0:5])

#get range of values for plots
plt.figure(figsize=(12,12))
plt.subplots_adjust(hspace = 0.4, wspace=0.5)
for i in range(0,8):
    for j in range(1,9):
        plt.subplot(8,8,8*i+j)
        try:
            plt.hist(chronology[8*i+j+1])
            plt.xlim(0,0.2)
        except:
            pass
plt.show

#try plotting over time
fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111)
ax.set_xlim(85,200)
ax.set_ylim(0,0.2)
for i in range(1,62):
    line=Line2D(chronology['Months'],chronology[i],color=cm.hsv(i/61))
    ax.add_line(line)
plt.show

#no overall trend over time.
#however, some topics seem to be periodic. 

for i in range(len(chronology)):
    chronology.at[i,'Month']=int(chronology.at[i,'Year-Month'][5:7])
#print(chronology[0:5])

chronology_monthsorted=chronology.sort_values('Month').loc[chronology['Months']>82]
for i in range(1,62):
    max=chronology_monthsorted[i].max()
    chronology_monthsorted[i]=chronology_monthsorted[i]/max
#justify normalising at this stage because I don't want the average to be overly influenced by particular years

#aggregate data by month only now
chronology_bymonth=pd.DataFrame(chronology_monthsorted['Month'].unique(),columns=['Month'])

for i in range(len(chronology_bymonth)):
    sliced_data=chronology_monthsorted.loc[chronology_monthsorted['Month']==chronology_bymonth.iat[i,0]]
    for j in range(1,62):
        chronology_bymonth.at[i,'n']=len(sliced_data[j])
        chronology_bymonth.at[i,j]=sliced_data[j].mean()

print(chronology_bymonth[0:5])
print(chronology_bymonth)

dec=pd.DataFrame(chronology_bymonth.loc[11]).transpose()
jan=pd.DataFrame(chronology_bymonth.loc[0]).transpose()

chronology_bymonth=pd.concat([dec,chronology_bymonth,jan]).reset_index()
chronology_bymonth=chronology_bymonth.reset_index().drop('index',1)
chronology_bymonth.at[0,'Month']=0
chronology_bymonth.at[13,'Month']=13

chronology_bymonth_extreme=chronology_bymonth.copy(deep=True)

mn=chronology_bymonth[i].min()
mx=chronology_bymonth[i].max()
(2*((chronology_bymonth[i]-mn)/mx))**2

for i in range(1,62):
    mn=chronology_bymonth[i].min()
    mx=chronology_bymonth[i].max()
    chronology_bymonth_extreme[i]=(2*((chronology_bymonth[i]-mn)/mx))**2
    
    
#try plotting over months instead
fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111)
ax.set_xlim(0,13)
ax.set_ylim(0,2.5)
ax.set_xlabel('Month')
ax.set_ylabel('Square of normalised weights')
ax.set_title('Topic weight by month (1 and 13 = Jan)')
for i in range(1,62):
    line=Line2D(chronology_bymonth_extreme['Month'],chronology_bymonth_extreme[i],color=cm.hsv(i/61))
    ax.add_line(line)
plt.show

#plot individually for easier comparison
for i in range(1,62):
    name='Topic '+str(i)+" - "+topic_list.iat[i-1,0]
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(111)
    ax.set_xlim(0.5,12.5)
    ax.set_ylim(0,1)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average weight')
    ax.set_title(name)
    line=Line2D(chronology_bymonth['Month'],chronology_bymonth[i],color=cm.hsv(i/61))
    ax.add_line(line)
    plt.show
    plt.savefig(name+'.png', dpi=fig.dpi)

"""Try clustering topics together based on which months of the year they appear"""

by_month_for_heirarchy=chronology_bymonth.drop(['Month','n'],1).transpose().drop([0,13],1).reset_index().drop('index',1)

#remove irrelevant topics: 
#the following topics are not relevant: 18, 34, 39, 54
#the following topics are language: 23 about school, 50 about clarification and questions, 55 MCQ choices, 56 file attachments
irrelevant=np.array([17,33,38,53,22,49,54,55])
by_month_for_heirarchy_cut=by_month_for_heirarchy.drop(irrelevant,0).reset_index().drop('index',1)
by_month_for_heirarchy_cut=by_month_for_heirarchy_cut.drop(0,0).reset_index().drop('index',1)
topic_list_cut=topic_list.drop(irrelevant,0).reset_index().drop('index',1)

from scipy.cluster.hierarchy import linkage, dendrogram

heirarchy = linkage(by_month_for_heirarchy_cut,'ward')

plt.figure(figsize=(50, 4))
plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('Topic')
plt.ylabel('distance')
dendrogram(
    heirarchy,
    color_threshold=0.7,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=12.,  # font size for the x axis labels
    labels=np.ravel(topic_list_cut)
)
plt.show()

from sklearn.cluster import KMeans

def clusterk(n):
    kmeans = KMeans(n_clusters=n, n_init=30, n_jobs=-1).fit(by_month_for_heirarchy_cut)
    labels=pd.DataFrame(kmeans.labels_)
    
    topic_list_cut.columns=['Topics']
    labels.columns=['Cluster']
    topics_kclustered=topic_list_cut.join(pd.DataFrame(labels)).sort_values('Cluster').reset_index()
    return(topics_kclustered)

clusterk(18)
