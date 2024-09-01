from access import mturk
from terminateHIT import stopHIT

response = mturk.list_hits(MaxResults=100)

for hits in response['HITs']:
    stopHIT(hits['HITId'])
    print(hits['HITId'], hits['HITStatus'])

print('all stopped')
