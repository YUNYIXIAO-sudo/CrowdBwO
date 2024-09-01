import boto3

MTURK = 'https://mturk-requester.us-east-1.amazonaws.com'
MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

mturk = boto3.client('mturk',

   aws_access_key_id = "WriteYourKeyIDHere",
   aws_secret_access_key = "WriteYourKeyHere",
   region_name='us-east-1',
   endpoint_url = MTURK_SANDBOX
)

print ("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my account")
'''activation:
$ cd AMTexperiment
$ virtualenv .
$ source bin/activate'''