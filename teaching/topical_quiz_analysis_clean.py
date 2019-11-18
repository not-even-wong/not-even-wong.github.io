import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 
studentresponses = pd.read_csv('student responses.csv')
studentresponses.drop(studentresponses.columns[studentresponses.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
rows=studentresponses.values.shape[0]
columns=studentresponses.values.shape[1]
print("No. of rows: "+str(rows))
print("No. of columns: "+str(columns))
headers=list(studentresponses.columns.values)
print("Column names: "+str(headers))
print("")
print('Ans key:')
anskey = pd.read_csv('sample ans key.csv', header=2)
anskey.drop(anskey.columns[anskey.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
anskey.drop('Qn',axis = 1, inplace = True)
rows=anskey.values.shape[0]
columns=anskey.values.shape[1]
print("No. of rows: "+str(rows))
print("No. of columns: "+str(columns))
headers=list(anskey.columns.values)
print("Column names: "+str(headers))
print("")
SR = anskey.loc[anskey['Ans'].isnull(), :]
sortedresponses = studentresponses.copy(deep=True)
rows=sortedresponses.values.shape[0]
columns=sortedresponses.values.shape[1]
print("")
print('Questions for quiz version selection are:')
print(SR[['Master']])
r=0
col=0
converter=pd.DataFrame([[1,2,3,4]], columns=['A','B','C','D'])
r=0
while r<rows:
    sortingrows = SR[['Master']]
    print('Students completed: '+str(r)+' / '+str(rows))
    while col<columns-2:    
        SRQ=sortingrows.iat[0,0]
        blankcheck=pd.isnull(studentresponses.iat[r,SRQ+1])
        while col<sortingrows.iat[0,0]-1:
            if blankcheck == False:
                SRQR=studentresponses.iat[r,SRQ+1]
                SRQRN=converter.at[0,SRQR]
                translated=int(anskey.iat[col,SRQRN])
                sortedresponses.iat[r,translated+1]=studentresponses.iat[r,col+2]
            else:
                sortedresponses.iat[r,col+2]=""
            col=col+1
        sortingrows.drop(sortingrows.index[0],inplace=True)
        col=col+1
    col=0
    r=r+1
print('Students completed: '+str(r)+' / '+str(rows))
print('')
sortedresponses.to_csv(r'output_sorted_responses.csv')
print('Sorted output saved')
print('')
print('Now performing question analysis')
print('')
answersummary = anskey.copy(deep=True)
answersummary.rename(columns={"Master":"Qn","Ver A":"A","Ver B":"B","Ver C":"C","Ver D":"D","Ans":"Correct"}, inplace=True)
answersummary['A']=0
answersummary['B']=0
answersummary['C']=0
answersummary['D']=0
answersummary['Correct']=0
c=0
while c<columns-2:
    r=0
    while r<rows:
        if sortedresponses.iat[r,c+2] == 'A':
            answersummary.at[c,'A'] = answersummary.at[c,'A']+1
        elif sortedresponses.iat[r,c+2] == 'B':
            answersummary.at[c,'B'] = answersummary.at[c,'B']+1
        elif sortedresponses.iat[r,c+2] == 'C':
            answersummary.at[c,'C'] = answersummary.at[c,'C']+1
        elif sortedresponses.iat[r,c+2] == 'D':
            answersummary.at[c,'D'] = answersummary.at[c,'D']+1
        if sortedresponses.iat[r,c+2] == anskey.at[c,'Ans']:
            answersummary.at[c,'Correct'] = answersummary.at[c,'Correct']+1
        r=r+1
    print('Questions analysed: '+str(c)+' / '+str(columns-3))
    c=c+1
print('')        
print(answersummary)
print('')
answersummary.to_csv(r'output_answer_summary.csv')
print('Answer summary saved')
print('')
print('Now analysing student grades')
print('')
studentmarked = sortedresponses.copy(deep=True)
r=0
while r<rows:
    c=2
    while c<columns:
        if sortedresponses.iat[r,c] == anskey.at[c-2,'Ans']:
            studentmarked.iat[r,c] = 1
        else:
            studentmarked.iat[r,c] = 0
        c=c+1
    r=r+1
print(studentmarked)
studentmarked.to_csv(r'output_students_marked.csv')
print('')
print('Student grade breakdown saved')
print('')
quizlist = SR[['Master']]
print('')
studentgrades = pd.DataFrame(np.zeros(shape=(len(studentmarked),len(quizlist)+2)))
studentgrades.rename(columns={0:'Class',1:'ID'}, inplace=True)
a=2
while a<len(quizlist)+2:
    colname = 'Quiz '+str(a-1)
    studentgrades.rename(columns={a:colname}, inplace=True)
    a=a+1
studentgrades['Class'] = studentmarked['Class']
studentgrades['ID'] = studentmarked['ID']
r=0
while r<rows:
    cgrade=0
    quizlistplaceholder=0
    cquiz=1
    while cgrade<len(quizlist):
        while cquiz<quizlist.iat[quizlistplaceholder,0]:
            studentgrades.iat[r,cgrade+2]=studentgrades.iat[r,cgrade+2]+studentmarked.iat[r,cquiz+1]
            cquiz=cquiz+1
        quizlistplaceholder=quizlistplaceholder+1
        cgrade=cgrade+1
    r=r+1
studentgrades.to_csv(r'output_student_grades.csv')
print('Student grade summary saved')
print('')
print('Calculating basic stats')
print('')
classlist=pd.Series(studentmarked['Class'].values.flatten()).unique()
print('List of '+str(len(classlist))+' classes: '+str(classlist))
print('')
classdata = pd.DataFrame(np.zeros(shape=(len(classlist),(6*len(quizlist))+1)))
classdata.rename(columns={0:'Class'}, inplace=True)
classdata['Class'] = classlist
a=1
cquiz=1
while a<(6*len(quizlist))+1:
    statcount=0
    statlist=pd.DataFrame([['mean','min','25th %','median','75th %','max']])
    while statcount<6:
        colname = statlist.at[0,statcount]+' '+str(cquiz)
        classdata.rename(columns={a:colname}, inplace=True)
        statcount+=1
        a=a+1
    cquiz+=1
classcount=0
while classcount<len(classlist):
    class_selection=pd.DataFrame(classlist).iat[classcount,0]
    classisolated=studentgrades.loc[studentgrades['Class']==class_selection].dropna()
    a=1
    cquiz=1
    while a<(6*len(quizlist))+1:
        quiz_selection='Quiz '+str(cquiz)
        classdata.iat[classcount,a]=round(classisolated[quiz_selection].mean(),2)
        a+=1
        classdata.iat[classcount,a]=round(classisolated[quiz_selection].min(),2)
        a+=1
        classdata.iat[classcount,a]=round(classisolated[quiz_selection].quantile(0.25),2)
        a+=1
        classdata.iat[classcount,a]=round(classisolated[quiz_selection].median(),2)
        a+=1
        classdata.iat[classcount,a]=round(classisolated[quiz_selection].quantile(0.75),2)
        a+=1
        classdata.iat[classcount,a]=round(classisolated[quiz_selection].max(),2)
        a+=1
        cquiz+=1
    classcount+=1
print('updated classdata:')
print(classdata)    
print('')
classdata.to_csv(r'output_class_statistics.csv')
print('Class statistics saved')
print('')
r=0
quizDI = pd.DataFrame(np.zeros(shape=(1,len(studentmarked)-2)))
c=0
while c<len(studentmarked)-2:
    questionname='Q'+str(c+1)
    quizDI.rename(columns={c:questionname}, inplace=True)
    c+=1
cgrade=0
cquiz=1
while cgrade<len(quizlist):
    quiz_selection='Quiz '+str(cgrade+1)
    quizmedian=studentgrades[quiz_selection].median()
    while cquiz<quizlist.iat[cgrade,0]:
        r=0
        while r<rows:
            currentgrade=studentgrades.iat[r,cgrade+2]
            if currentgrade>quizmedian:
                quizDI.iat[0,cquiz-1]+=studentmarked.iat[r,cquiz+1]
            elif currentgrade<quizmedian:
                quizDI.iat[0,cquiz-1]-=studentmarked.iat[r,cquiz+1]
            r+=1
        quizDI.iat[0,cquiz-1]=round(quizDI.iat[0,cquiz-1]/rows,2)
        cquiz=cquiz+1
    cgrade=cgrade+1
print('Discrimination index: ')
print(quizDI)
quizDI.to_csv(r'output_quiz_DI.csv')
print('Discrimination index saved')
print('')