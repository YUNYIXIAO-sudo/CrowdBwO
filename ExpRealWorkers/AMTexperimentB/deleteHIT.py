from access import mturk
import pandas as pd

response = mturk.list_hits(MaxResults=100)

for hits in response['HITs']:
    print(hits['HITId'])
    
    notation = mturk.get_hit(HITId=hits['HITId'])['HIT']['RequesterAnnotation']
    
    assign = mturk.list_assignments_for_hit(
        HITId=hits['HITId'],
        MaxResults=100,
    )

    for a in assign['Assignments']:
        a = {key: [value] for key, value in a.items()}
        a['score'] = notation
        df = pd.DataFrame(a)
        df.to_csv('assignmentResults0319.csv', mode='a', index=False)

    
    response = mturk.delete_hit(
        HITId=hits['HITId']
    )
    
    
    
print('all deleted')