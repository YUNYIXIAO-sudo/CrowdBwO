from access import mturk
import xmltodict
import numpy as np
from parameter import para
import pandas as pd
from datetime import datetime
import time

df = pd.DataFrame(columns=['assignmentId'])
df.to_csv('AssignmentRecs' + str(datetime.now().day) + '.csv', mode='w', index=False)
df = pd.DataFrame(columns=['workers','hits','score','submitTime','answer0','answer1','answer2','status','assignmentId'])
df.to_csv('AssignmentResults' + str(datetime.now().day) + '.csv', mode='w', index=False)

KeyQues = '0'

answer_key = { 
    '0': '350',
    '1': 'renewable',
    '2': 'exciting'
}


def analyzeResults(qualification):
    collectedAnswers = np.zeros(2)
    correctAnswers = np.zeros(2)

    worker = []
    hitids = []
    scores = []
    submitTimes = []
    answers0 = []
    answers1 = []
    answers2 = []
    status = []
    assignmentId = []



    hits = mturk.list_hits(MaxResults=100)

    for hit in hits['HITs']:
        createTime = str(hit['CreationTime']).split(' ')[0]
        
        if createTime == '2024-03-21':  #the publish day

            #if qualiRes['HIT']['QualificationRequirements'][0]['QualificationTypeId'] == qualification:
            tasks = mturk.get_hit(HITId=hit['HITId'])['HIT']
            taskNotation = tasks['RequesterAnnotation']

            if taskNotation == 'stask1' or 'stask2':
            
                assignments = mturk.list_assignments_for_hit(
                    HITId=hit['HITId'],
                    MaxResults=100
                )
        

                for assignment in assignments['Assignments']:

                    df = pd.read_csv('AssignmentRecs' + str(datetime.now().day) + '.csv')
                    assignRec = df['assignmentId']

                    if assignment['AssignmentId'] not in list(assignRec):
                        answerxml = assignment['Answer']
                        doc = xmltodict.parse(answerxml)
                        Answers = doc.get('QuestionFormAnswers', {}).get('Answer', [])
                        quesAnswer = [answer['SelectionIdentifier'] for answer in Answers if 'SelectionIdentifier' in answer]
                    
                        
                        worker.append(assignment['WorkerId'])
                        hitids.append(hit['HITId'])
                        scores.append(str(taskNotation))
                        submitTimes.append(str(assignment['SubmitTime']))
                        answers0.append(quesAnswer[0])
                        answers1.append(quesAnswer[1])
                        answers2.append(quesAnswer[2])
                        status.append(assignment['AssignmentStatus'])
                        assignmentId.append(assignment['AssignmentId'])

                        df = pd.DataFrame({'assignmentId': [assignment['AssignmentId']]})
                        df.to_csv('AssignmentRecs' + str(datetime.now().day) + '.csv', mode='a', index=False, header=False)


    df = pd.DataFrame({
        'workers': worker,
        'hits': hitids,
        'score': scores,
        'submitTime': submitTimes,
        'answer0': answers0,
        'answer1': answers1,
        'answer2': answers2,
        'status': status,
        'assignmentId': assignmentId
    })

    quaName = qualification[:5]
    

    df.to_csv('AssignmentResults' + str(datetime.now().day) + '.csv', mode='a', header=False, index=False)

    ansdf = pd.read_csv('AssignmentResults' + str(datetime.now().day) + '.csv')
    ans = np.array(ansdf['answer' + KeyQues])
    score = np.array(ansdf['score'])


    for i in range(len(ans)):
        if score[i] == 'stask1':
            collectedAnswers[0] += 1
            if str(ans[i]) == answer_key[KeyQues]:
                correctAnswers[0] += 1

            
        elif score[i] == 'stask2':
            collectedAnswers[1] += 1
            if str(ans[i]) == answer_key[KeyQues]:
                correctAnswers[1] += 1


    return collectedAnswers, correctAnswers


a, b = analyzeResults('')


