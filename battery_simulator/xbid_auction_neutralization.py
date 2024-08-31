import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import time
import pprint
import math
import openpyxl
from openpyxl import Workbook, load_workbook

from xbid_data_getter import xbid_data

from battery_simulator import Position_xbid
from allocation_optimizer import *

from xbid_allocation_live import *


def get_current_xbid_price(start, end, type=None):
    # returns the average price in the xbid, (usage: for the following hour)
    bid, ask = xbid_data().get_best_bid_ask_profile(start, end)
    ask, bid = np.fromiter(ask.values(), dtype=float), np.fromiter(bid.values(), dtype=float)
    if type == 'B':
        return np.mean(bid)
    else:
        return np.mean(ask)


class outer_spread:

    def __init__(self, start_index: dt.datetime, end_index: dt.datetime, margin=None):
        self.start_index = start_index
        self.end_index = end_index
        self.margin = margin


class inner_spread:

    def __init__(self, start_index: dt.datetime, end_index: dt.datetime, margin=None):
        self.start_index = start_index
        self.end_index = end_index
        self.margin = margin


class auction_xbid:

    """
    We get the auction subsection schedule in a dataframe, and generate all the required parameters based on that dataframe
    """

    def __init__(self, subsection):
        self.subsection = subsection  # has columns: da, id, type
        self.start_time = subsection.index[0]
        self.end_time = subsection.index[-1] + dt.timedelta(hours=1)
        self.begin = None

    def get_subsection_actions(self):
        da_action = []
        id_action = []
        for i in range(len(self.subsection)):
            if self.subsection.da[i] != 0 and self.subsection.type[i] == 'B':
                margin = self.subsection.xbid_price[i] - self.subsection.xbid_price[-1]
                action = (self.start_time, self.end_time, margin)
                da_action.append(action)
            elif self.subsection.id[i] != 0:
                if self.subsection.type[i] == 'B':
                    margin = self.subsection.xbid_price[i] - self.subsection.xbid_price[i+i]
                else:
                    margin = - self.subsection.xbid_price[i] + self.subsection.xbid_price[i+i]
                action = (self.subsection.index[i], self.subsection.index[i+1], margin)
                id_action.append(action)
                i += 1

        return da_action, id_action

    def get_current_timestamp(self):
        if dt.datetime.utcnow() >= self.start_time:
            current_timestamp = dt.datetime.utcnow()
        else:
            current_timestamp = self.start_time

        return current_timestamp







