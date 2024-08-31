import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
from xbid_data_getter import xbid_data
from battery_simulator import Position_xbid
from allocation_optimizer import *


time_from = dt.datetime.utcnow() + dt.timedelta(minutes=10)
tim_to = time_from + dt.timedelta(hours=6)


class xbid_data_inpsector:

    def __init__(self):
        self.time_from = dt.datetime.utcnow() + dt.timedelta(minutes=10)
        self.time_to = time_from + dt.timedelta(hours=24) + dt.timedelta(days=1)
        self.xbid_data = xbid_data()

    def update_time_bounds(self):
        self.time_from = dt.datetime.utcnow() + dt.timedelta(minutes=10)
        self.time_to = time_from + dt.timedelta(hours=6)

    def run(self):
        x= xbid_data.scrap()
        contracts = self.xbid_data.get_contracts(self.time_from, self.time_to)
        return x


a = xbid_data_inpsector()
scrap = a.run()
x=2



