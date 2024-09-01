from access import mturk

hits = mturk.list_hits(MaxResults=100)

for hit in hits['HITs']:
    #if hit['HITReviewStatus'] == 'NotReviewed':
    print('HITID:', hit['HITId'])
    assignments = mturk.list_assignments_for_hit(
        HITId=hit['HITId']
    )
    for assignment in assignments['Assignments']:
        print('ASSIGNMENT ID:',assignment['AssignmentId'], assignment['AssignmentStatus'])
        '''response = mturk.approve_assignment(
            AssignmentId=assignment['AssignmentId']
        )'''

print('approved')
