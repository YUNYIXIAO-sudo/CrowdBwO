import xmltodict
from access import mturk


answerkey = open('qualification_task_answer.xml', mode='r').read()

qualificationQuestion = open('qualification_task_design_eng.xml', mode='r').read()


response1 = mturk.create_qualification_type(
        Name='Qualification0126ta',
        Description='Qualification Question 240126a',
        QualificationTypeStatus='Active',
        Test=qualificationQuestion,
        AnswerKey=answerkey,
        TestDurationInSeconds=300
    )
    
qualification1 = response1['QualificationType']['QualificationTypeId']

response2 = mturk.create_qualification_type(
        Name='Qualification0126tb',
        Description='Qualification Question 240126b',
        QualificationTypeStatus='Active',
        Test=qualificationQuestion,
        AnswerKey=answerkey,
        TestDurationInSeconds=300
    )
    
qualification2 = response2['QualificationType']['QualificationTypeId']

print(qualification1, qualification2)

# mturk qualification id
# qualificationId = '3PP3A267ACWSW254AD0659SN9OC9Q4' '329CD4N5N34KYNLSVTXFN8ZH2RZZBX'

#sandboxid '37U3UQN45GEJ33WUUS2249SYY5RCLV' '332K4KOFDLOGBZDM5VBUQVISG7S0DA'

qualificationId = ''

'''
def updateQualification():

    response = mturk.update_qualification_type(
        QualificationTypeId=qualificationId,
        Description='Qualification Question 231128',
        QualificationTypeStatus='Active',
        Test=qualificationQuestion,
        AnswerKey=answerkey,
        TestDurationInSeconds=600
    )

    print('updated')

updateQualification()
'''
