from access import mturk

question = open('task_design_eng.xml', mode='r').read()


def publishHIT2(qualification_type_id):

    # Create a HIT with the age qualification
    new_hit = mturk.create_hit(
        Title='(~ 1 Minute) Simple Questions 20240322 (HIT Approval Rate >= 98%; approved HITs >= 500)',
        Description='There are 3 questions given to you. Select the answers according to your knowledge.',
        Keywords='Easy, quick',
        Reward='0.02',
        MaxAssignments=100,
        AutoApprovalDelayInSeconds=600,
        LifetimeInSeconds=43200, #change time when switch settings
        AssignmentDurationInSeconds=600,
        Question=question,
        QualificationRequirements=[
            {
                'QualificationTypeId': '000000000000000000L0', # Approval rate
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues':[98],
                'ActionsGuarded': 'PreviewAndAccept'
            },{
                'QualificationTypeId': '00000000000000000040', # Approved number
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues':[500],
                'ActionsGuarded': 'PreviewAndAccept'
            },{
                'QualificationTypeId':"00000000000000000071",
                'Comparator':"EqualTo",
                'LocaleValues':[{
                    'Country':"US"}],
                'ActionsGuarded': 'PreviewAndAccept'
            }
        ],
        RequesterAnnotation='task2'
    )




    hitId = new_hit['HIT']['HITId']

    print ("A new HIT2 has been created. You can preview it here:")
    print('https://www.mturk.com/mturk/preview?groupId=' + new_hit['HIT']['HITGroupId'])
    #print ("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    print('HIT2 ID: ' + hitId)

    return hitId




