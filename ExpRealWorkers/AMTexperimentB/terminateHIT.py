from access import mturk
import datetime


def stopHIT(hitid):
    mturk.update_expiration_for_hit(
        HITId=hitid,
        ExpireAt=datetime.datetime(2023, 12, 20)
    )

    print(str(hitid) + ' STOPPED')