from access import mturk

hits = mturk.list_hits(MaxResults=100)

'''for hit in hits['HITs']:
    assignments = mturk.list_assignments_for_hit(
        HITId=hit['HITId'],
        MaxResults=100
    )

    for assignment in assignments['Assignments']:
        response = mturk.delete_worker_block(
            WorkerId=assignment['WorkerId']
        )'''

block = mturk.list_worker_blocks(MaxResults=100)

while len(block['WorkerBlocks']) > 0:
    for b in block['WorkerBlocks']:
        print(b['WorkerId'])
        response = mturk.delete_worker_block(
                WorkerId=b['WorkerId']
            )
    print('finish')
    block = mturk.list_worker_blocks(MaxResults=100)

print('block released')