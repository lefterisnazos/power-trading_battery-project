from xbid_allocation_live import *
from production import *
import heapq
import logging
import pytz


class Environment:

    def __init__(self):
        self.date = dt.date.today()
        self.datetime_date = dt.datetime.today()
        self.current_timestamp = dt.datetime.utcnow()

    def update(self):
        self.date = dt.date.today()
        self.datetime_date = dt.datetime.today()
        self.current_timestamp = dt.datetime.utcnow()

    @property
    def time_shift(self):

        """
        we are using utc+00 times inside the xbid_automator.run() function. But our reference is CET time. So we adjust accourdingly based on timeshift
        if DST is on difference is 1 hour, if DST is off difference is 2 hours
        :return:
        """
        timeZone = pytz.timezone("Europe/Berlin")
        time_now = dt.datetime.utcnow()
        if timeZone.localize(time_now).dst() == dt.timedelta(0, 0):
            return 1
        else:
            return 2


class xbid_automator:

    def __init__(self):
        self.new_schedule_time = None
        self.current_schedule = None
        self.instances_list = []
        self.instances_activated_list = []

    def run(self):
        env = Environment()
        executor = ThreadPoolExecutor()
        self.new_schedule_time = env.datetime_date.replace(hour=3 - env.time_shift, minute=30, second=0)

        while True:
            env.update()
            if not self.instances_list or env.datetime_date > self.new_schedule_time:
                self.current_schedule = xbid_scheduler(self.new_schedule_time.date())
                self.current_schedule.get_xbid_instances(logging_on=False)
                self.new_schedule_time = env.datetime_date.replace(hour=3 - env.time_shift, minute=15, second=0) + dt.timedelta(days=1)
                for instance in self.current_schedule.xbid_instances.copy():
                    # append instances, even if they havent started yet, when ought to be, up to 2 hours, ignore others, del them (del self.current_schedule)
                    if instance.start_time > dt.datetime.utcnow() - dt.timedelta(hours=1):
                        self.instances_list.append(instance)
            if self.instances_list:
                for instance in self.instances_list:
                    if dt.datetime.utcnow() > instance.start_time - dt.timedelta(hours=1, minutes=0):
                        self.instances_activated_list.append(self.instances_list.pop(0))
                        #  self.instances_activated_list[-1].run()
                        self.instances_activated_list.append(executor.submit(self.instances_activated_list[-1].run(),))

            time.sleep(60)


class xbid_scheduler:

    def __init__(self, date):
        self.date = date
        self.next_day = date + dt.timedelta(days=1)
        self.datetime_end_date = dt.datetime(date.year, date.month, self.next_day.day)

        self.strat = combined_strategy(date, date)
        self.strat.get_schedule()
        self.schedule = self.strat.scheduling
        self.schedule = self.schedule.tz_localize('CET').tz_convert('UTC')
        try:
            self.revenue_forecast = self.strat.get_revenue_forecast()
        except Exception as e:
            print('revenue forecast error ')

        self.time_threshold = 60 * 1.25  # measured in seconds
        self.bid_times = {'fcr': self.datetime_end_date.replace(hour=6, minute=30), 'da': self.datetime_end_date.replace(hour=12, minute=0), 'id': self.datetime_end_date.replace(hour=12, minute=0)}
        self.xbid_instances = []

    def get_xbid_instances(self, logging_on=False, action_pair_cost=38):

        intervals = self.get_xbid_intervals_tz_aware()
        # possible state levels, 0(empty) or 4(filled).
        for interval in intervals:
            xbid_instance = xbid_strategy(interval[0].tz_localize(None), interval[1].tz_localize(None),
                                          interval[2], interval[2], quantity=interval[3], action_pair_cost=action_pair_cost,
                                          logging_ON=logging_on, called_by_automator=True)
            self.xbid_instances.append(xbid_instance)

        return self.xbid_instances

    def get_xbid_intervals_tz_aware(self):
        df = self.schedule.copy()
        start_times = []
        end_times = []
        quantities = []
        starting_states = []  # possible states are 0 and 4,for fixed quantities on the allocator
        if df.da.iloc[0] == 0:
            start_times.append(df.index[0])
            quantities.append(df.xbid.iloc[0])
            starting_states.append(0)
        for i in range(len(df) - 1):
            if df.da.iloc[i] == 0 and i < len(df) - 1:
                continue
            if df.da.iloc[i][-1] == 'B':
                start_times.append(df.index[i+1])
                quantities.append(int(df.xbid.iloc[i+1]))
                starting_states.append(4)
                if start_times and df.da.iloc[i] != 0:
                    end_times.append(df.index[i])
            elif df.da.iloc[i][-1] == 'S':
                end_times.append(df.index[i])
                try:
                    if df.da.iloc[i+1] == 0:
                        start_times.append(df.index[i+1])
                        quantities.append(int(df.xbid.iloc[i+1]))
                        starting_states.append(0)
                except Exception as e:
                    continue
        start_times = list(dict.fromkeys(start_times))
        if len(end_times) < len(start_times):
            # we want to get up to 00am in CET TIME
            # we use the left bracket in indeces, so given CET reference, the last index in utc time will either be 21:00(dst=on) orr 22:00(dst=off)
            end_times.append(df.index[-1] + dt.timedelta(hours=df.index[-1].tz_convert('CET').hour - df.index[-1].hour))
        intervals = list(zip(start_times, end_times, starting_states, quantities))

        return intervals


x = xbid_automator()
b = x.run()
y=2