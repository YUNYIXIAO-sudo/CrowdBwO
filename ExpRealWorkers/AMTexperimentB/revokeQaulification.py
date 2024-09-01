from access import mturk


def revokeWorkers1(hitid, qualification):
    assignments = mturk.list_assignments_for_hit(
        HITId=hitid,
        MaxResults=100
    )
    for assignment in assignments['Assignments']:
        block = mturk.create_worker_block(
            WorkerId=assignment['WorkerId'],
            Reason='temporary block to avoid multiple assignments'
        )

    '''for assignment in assignments['Assignments']:
        response = mturk.associate_qualification_with_worker(
            QualificationTypeId=qualification,
            WorkerId=assignment['WorkerId'],
            IntegerValue=3
        )'''


def revokeWorkers2(hitid, qualification):
    assignments = mturk.list_assignments_for_hit(
        HITId=hitid,
        MaxResults=100
    )

    for assignment in assignments['Assignments']:
        block = mturk.create_worker_block(
            WorkerId=assignment['WorkerId'],
            Reason='temporary block to avoid multiple assignments'
        )
