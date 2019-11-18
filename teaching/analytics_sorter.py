import statistics as s
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None) 

"""
Key in key info here: 
    1) choose number of groups
    2) choose which file to use
    
"""

n_groups = 5
student_responses = pd.read_csv('Sample Responses 2.csv')
answer_key = pd.read_csv('Sample Solutions.csv', header=None)

#define parameters
n_questions=len(answer_key)
n_students=len(student_responses)
student_marked=pd.DataFrame(np.zeros(shape=(n_students,3*n_questions+2),dtype='|U16'))

#mark student response: create new table
#populate so there are 3 columns for each question, corresponding to answer, confidence, right/wrong
#last two columns for ID and group

r=0
while r<n_students:
    c=0
    while c<n_questions:
        student_marked.iat[r,c*3]=student_responses.iat[r,(c*2)+1]
        student_marked.iat[r,(c*3)+1]=student_responses.iat[r,(c*2)+2]
        if student_responses.iat[r,(c*2)+1]==answer_key.iat[c,1]:
            student_marked.iat[r,(c*3)+2]=1
        else:
            student_marked.iat[r,(c*3)+2]=0
        c+=1
    student_marked.iat[r,c*3]=student_responses.iat[r,(c*2)+1] #this is their ID number
    student_marked.iat[r,c*3+1]=999 #this is their group number
    r+=1

#create summary: new table with one column for each question, 2 columns, 
#0 contains qn number and 1 contains number of correct responses for that question
ans_summary=pd.DataFrame(np.zeros(shape=(n_questions,2)))
r=0
while r < n_questions:
    ans_summary.iat[r,0]=r
    r1=0
    while r1 < n_students:
        ans_summary.iat[r,1]+=student_marked.iat[r1,(r*3+2)]
        r1+=1
    r+=1

#sort summary based on number of correct responses to that question
ans_summary=ans_summary.sort_values(1).reset_index(drop=True)

#rearrange responses: duplicate student mark table, populate in order based on rearranged summary

student_sorted=student_marked.copy(deep=True)
c=0
while c<len(ans_summary):
    Q=ans_summary.iat[c,0]
    #print('Question '+str(Q)+' is in position '+str(c))
    
    student_sorted[3*c]=student_marked[3*Q]
    student_sorted[3*c+1]=student_marked[3*Q+1]
    student_sorted[3*c+2]=student_marked[3*Q+2]
    c+=1

ans_summary.rename(columns={0: 'Qn',1: 'Correct'},inplace=True)
print()
print(ans_summary)
print()

#define max size as N/n rounded up
group_size = round(0.5+n_students/n_groups)
print('Sort '+str(n_students)+' students into '+str(n_groups)+' groups of '+str(group_size))
print()

#create table for tracking group info: 
#group number, number of students per group, number of correct for each question
group_info=pd.DataFrame(np.zeros(shape=(n_groups,2+n_questions)))
group_info_summary=pd.DataFrame(np.zeros(shape=(2,2+n_questions)))
r=0
while r<n_groups:
    group_info.iat[r,0]=r
    r+=1

#sorting students into groups
"""
loop for each question:
    Identify students who are correct
    for each correct student:
        Sort groups first by total number, then by number of correct responses
        Add this student to the top row 
        (i.e. out of all groups with the least number of correct responses, the group with the least students)
    rearrange students based on answer to current question 
    (so subsequent additions based on next qn likely to have an even spread for this question)
    move to next Qn
"""

Q=1
while Q<n_questions+1:
    currentQgrade=3*Q-1
    r=0
    while r<n_students:
        group_info=group_info.sort_values(1).reset_index(drop=True)
        group_info=group_info.sort_values(Q+1).reset_index(drop=True)
        if student_sorted.iat[r,currentQgrade]==1:
            if student_sorted.iat[r,3*n_questions+1]==999:
                student_sorted.iat[r,3*n_questions+1]=group_info.iat[0,0]
                group_info.iat[0,1]+=1
                i=Q
                while i<n_questions+1:
                    group_info.iat[0,i+1]+=student_sorted.iat[r,3*i-1]
                    i+=1
        r+=1
    i=1
    while i<n_questions+2:
        group_info_summary.iat[0,i]=group_info[i].sum()
        i+=1
    student_sorted=student_sorted.sort_values(3*(Q-1))
    group_info=group_info.sort_values(0).reset_index(drop=True)
    Q+=1

#adding in the last students who didn't get the last qn right
Q-=1
group_info=group_info.sort_values(1).reset_index(drop=True)
group_info=group_info.sort_values(Q+1).reset_index(drop=True)    
r=0
while r<n_students: 
    if student_sorted.iat[r,3*n_questions+1]==999:
        student_sorted.iat[r,3*n_questions+1]=group_info.iat[0,0]
        group_info.iat[0,1]+=1
        group_info_summary.iat[0,1]+=1
        i=Q
        while i<n_questions:
            group_info.iat[0,i+1]+=student_sorted.iat[r,3*i-1]
            i+=1
    r+=1

#arrange students by register number within group
student_sorted=student_sorted.sort_values(3*n_questions).reset_index(drop=True)
student_sorted=student_sorted.sort_values(3*n_questions+1).reset_index(drop=True)
group_info=group_info.sort_values(0).reset_index(drop=True)

#check summary
r=1
group_info_summary.iat[1,r]=len(student_sorted)
while r<n_questions+1:
    group_info_summary.iat[1,r+1]=student_sorted[3*r-1].sum()
    r+=1
    
#show % correct in each group
group_info_percentages=group_info.copy(deep=True)
r=0
while r<n_groups:
    c=2
    while c<n_questions+2:
        group_info_percentages.iat[r,c]=round(group_info.iat[r,c]/group_info.iat[r,1],2)
        c+=1
    r+=1

#show final output
print(student_sorted[[3*n_questions,3*n_questions+1]])
print()
print(group_info)
print(group_info_summary)
print()
print('Percentage in each group who got this question correct:')
#print(group_info_percentages)

