from hitApproval1 import publishHIT1
from hitApproval2 import publishHIT2
from terminateHIT import stopHIT
from answers import analyzeResults   
from access import mturk
import time
import numpy as np
from revokeQaulification import revokeWorkers1, revokeWorkers2

qualification = '332K4KOFDLOGBZDM5VBUQVISG7S0DA'

"""
response = mturk.disassociate_qualification_from_worker(
    WorkerId='A3FMLV7U8J5YYB',
    QualificationTypeId=qualification
)
print('revoked')




response = mturk.associate_qualification_with_worker(
    QualificationTypeId=qualification,
    WorkerId='A3FMLV7U8J5YYB',
    IntegerValue=2
)
print('done')
"""

hitid1 = publishHIT1(qualification)
hitid2 = publishHIT2(qualification)

print(hitid1, hitid2)

time.sleep(180)

stopHIT(hitid1)
stopHIT(hitid2)
#revokeWorkers1(hitid1, qualification)
#revokeWorkers2(hitid2, qualification)

results = analyzeResults(qualification)

print(results)