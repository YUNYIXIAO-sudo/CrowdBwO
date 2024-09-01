from access import mturk
import pandas as pd

days = ['11', '11', '12', '12']
qualifications = ['3VOIX1GCVU7JAJSPATHRZQ0F0LHYS6', '36E1FXTU1O7N3R0KNIONVEF5PCN1BB', '3PDP3C7SCEYYM8C5F3T6KJO9QOH3VB', '352JGGUH30ADG8C0SKAJE46Q4AF1FT']

for i in range(len(qualifications)):

    workerids = []
    workerScores = []
    qualidays = []

    response = mturk.list_workers_with_qualification_type(
        QualificationTypeId=qualifications[i],
        MaxResults=100
    )

    for worker in response['Qualifications']:
        workerid = worker['WorkerId']
        workerScore = worker['IntegerValue']

        workerids.append(workerid)
        workerScores.append(workerScore)
        qualidays.append(days[i])
    

    df = pd.DataFrame({
        'worker': workerids,
        'score': workerScores,
        'taskDay': qualidays
    })

    df.to_csv('workersbyQuali.csv', mode='a')






