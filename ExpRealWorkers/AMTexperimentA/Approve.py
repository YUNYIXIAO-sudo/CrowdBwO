from access import mturk

hits = mturk.list_hits(MaxResults=100)

for hit in hits['HITs']:
    if hit['HITReviewStatus'] == 'NotReviewed':
        print('HITID:', hit['HITId'], hit['CreationTime'], hit['RequesterAnnotation'])
        
        assignments = mturk.list_assignments_for_hit(
            HITId=hit['HITId'],
            MaxResults=100
        )
        for assignment in assignments['Assignments']:
            print('ASSIGNMENT ID:',assignment['AssignmentId'], assignment['AssignmentStatus'])
            if assignment['AssignmentStatus'] == 'Submitted':
                response = mturk.approve_assignment(
                    AssignmentId=assignment['AssignmentId']
                )

print('approved')
