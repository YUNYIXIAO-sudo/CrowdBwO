from access import mturk
import xmltodict
import numpy as np
from parameter import para
import pandas as pd
from datetime import datetime


qualification = '3JFN40I01XKKZGM9G47LF638NNHMB3'
#'30TX348SI463PB3WLT6QLY0IMN758T'


f = open('AssignmetnResults' + str(datetime.now()) + '.csv', 'a')
f.write('worker, score, time, answer1, answer2, answer3, status \n')

def printFinalResults():
    hits = mturk.list_hits()

    for hit in hits['HITs']:
        if hit['Title'] == 'Simple Questions':
            print('HITID:', hit['HITId'])
            assignments = mturk.list_assignments_for_hit(
                HITId=hit['HITId']
            )

            for assignment in assignments['Assignments']:
                qualify = mturk.get_qualification_score(
                    QualificationTypeId=qualification,
                    WorkerId=assignment['WorkerId']
                )
                score = qualify['Qualification']['IntegerValue']

                answerxml = assignment['Answer']
                doc = xmltodict.parse(answerxml)
                Answers = doc.get('QuestionFormAnswers', {}).get('Answer', [])
                quesAnswer = [answer['SelectionIdentifier'] for answer in Answers if 'SelectionIdentifier' in answer]
            
                f.write(assignment['WorkerId'] + ',' + str(score) + ',' + str(assignment['SubmitTime']) + ',' + quesAnswer[0] + ',' + quesAnswer[1] + ',' + quesAnswer[2] + ',' + assignment['AssignmentStatus'] + '\n')


