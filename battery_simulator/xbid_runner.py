from xbid_allocation_live import *


time_from = dt.datetime(2023, 6, 21, 22, 0)
time_to = dt.datetime(2023, 6, 22, 22, 0)
state = 0

while dt.datetime.utcnow() < time_from - dt.timedelta(hours=2, minutes=30):
    time.sleep(60*10)
a = xbid_strategy(time_from, time_to, state, state, action_pair_cost=37)
run_results = a.run()

print(run_results)


