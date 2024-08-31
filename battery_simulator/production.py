import pandas as pd
import numpy as np
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import time
import pprint
import math
import openpyxl
import itertools
import json
from battery_simulator import Battery, Position, Position_xbid
from enscohelper import date_time
from enscohelper.database import DBConnector
from enscohelper.datafeed import ActualData, ForecastData, FCRData, EnscoData
from sklearn.preprocessing import StandardScaler
from fcr_vs_da_production import fcr_vs_da_production, day_ahead_production
from auction_15min_production import id_vs_da_production, consecutive_integer_check, add_weekday_hour_feature
from xbid_auction_neutralization import *
import threading
from itertools import chain


def take_integer(x):
    try:
        x = int(x[:-1])
    except:
        return 0
    if isinstance(x, int):
        return x
    else:
        return 0


class combined_strategy:
    """
    1) First choose whether to allocate more mw on FCR or in DA
    2) Decide on the hours for intraday_15min_auction-DA spread
    3) Run intraday_continuous strategy for all the remaining hours
    """
    def __init__(self, date_from, date_to, id_days_back=7, id_margin_barrier=2500, technical_availability_constraint=False):

        self.date_from = date_from
        self.date_to = date_to
        self.date = self.date_from
        self.scheduling = pd.DataFrame(columns=['fcr', 'da', 'id', 'xbid'], index=pd.date_range(self.date, self.date + dt.timedelta(days=1), freq='1H', closed='left'))
        self.actual_prices = None

        # the fcr_vd_da & id_vs_da prod numbers are based on the forecast values
        self.fcr_vs_da = fcr_vs_da_production(self.date)
        self.id_vs_da = id_vs_da_production(self.date, days_back=id_days_back, margin_barrier=id_margin_barrier) # minimum days_back=1, days_back =0 doesnt work because it takes the todays date
        self.total_quantity = 10
        self.id_da_quantity = 3

        self.get_schedule()

        if not technical_availability_constraint:
            self.fcr_allocation_elveton = pd.read_excel('fcr_allocation_elveton.xlsx', index_col=0).fillna(0)
        else:
            self.fcr_allocation_elveton = pd.read_excel('fcr_allocation_technical_availability.xlsx', index_col=0)

    def fcr_allocation_elveton_transformation_by_date(self):
        pass

    def xbid_strategy(self, start: dt.datetime, end: dt.datetime):
        """
        * run the intraday_continuous strategy for all the remaining hours
        """
        xbid = xbid_strategy(start, end)
        xbid.run()
        # create print function, that when allocation gets updated, it prints a dictionary with keys and values or moves, with the moves_needed to make (
        # have to compare old and new allocation and take difference

        return 0

    def get_schedule(self, include_id=True):

        """
        scheduling datetime index in CET mode
        """
        allocation_fcr_vs_da = self.fcr_vs_da.get_allocation_results()
        allocation_id_vs_da = self.id_vs_da.get_allocation_results(quantity=self.id_da_quantity)
        allocation_fcr_vs_da, allocation_id_vs_da = allocation_fcr_vs_da.fillna(0), allocation_id_vs_da.fillna(0)

        for i in range(len(self.scheduling)):
            self.scheduling['fcr'][i] = allocation_fcr_vs_da.fcr[i]
            self.scheduling['id'][i] = allocation_id_vs_da.id[i]
            if allocation_fcr_vs_da.da[i] == 0 and allocation_id_vs_da.da[i] == 0:
                self.scheduling['da'][i] = 0
            elif allocation_fcr_vs_da.da[i] == 0:
                self.scheduling['da'][i] = allocation_id_vs_da.da[i]
            else:
                self.scheduling['da'][i] = allocation_fcr_vs_da.da[i]
        if include_id:
            self.scheduling['xbid'] = [self.total_quantity]*len(self.scheduling) - self.scheduling.fcr - self.scheduling.da.map(take_integer) - self.scheduling.id.map(take_integer)
        else:
            self.scheduling['xbid'] = [self.total_quantity] * len(self.scheduling) - self.scheduling.fcr - self.scheduling.da.map(take_integer)

    def get_revenue_forecast(self, inspection=True):

        # fcr_da revenue part
        revenue_fcr_vs_da = self.fcr_vs_da.get_revenue_forecast()
        # da revenue part
        day_ahead_prod = day_ahead_production(self.date_from, self.date_to)
        da_forecast = day_ahead_prod.get_forecast_prices()
        da_allocation_without_id = self.scheduling.copy()
        da_rev = 0
        for i in range(len(da_allocation_without_id)):
            if da_allocation_without_id.da[i] != 0:
                if str(da_allocation_without_id.da[i])[-1] == 'B':
                    da_rev = da_rev - da_forecast.price[i] * int(str(da_allocation_without_id.da[i])[:-1])
                else:
                    da_rev = da_rev + da_forecast.price[i] * int(str(da_allocation_without_id.da[i])[:-1])

        fcr_rev = revenue_fcr_vs_da[0].revenue.sum()
        if inspection is True:
            return fcr_rev,  da_rev
        else:
            return revenue_fcr_vs_da, fcr_rev, da_rev

    def get_bid_proposals(self):
        # run it before 1pm cet
        da_bid, fcr_bid = self.fcr_vs_da.get_bid_proposals()
        id_bid = self.id_vs_da.get_bid_proposals()

        return da_bid, fcr_bid, id_bid

    def get_revenue_actual(self, strategy_by_strategy_calculation=True, fcr_elveton=False):
        da_actual, fcr_actual = self.fcr_vs_da.get_actual_prices()
        id_actual = self.id_vs_da.get_actual_prices()
        fcr_rev, da_rev, id_rev = 0, 0, 0
        fcr_allocation = self.scheduling.iloc[:, [0]].resample('4H', label='left').agg({'fcr': 'first'})
        da_allocation = self.scheduling.iloc[:, [1]]
        id_allocation = self.scheduling.iloc[:, [2]]

        da_zero_rev = False
        if fcr_elveton:
            da_allocation_temp = da_allocation.copy()
            allocation_elveton = self.fcr_allocation_elveton[self.fcr_allocation_elveton.index.date == self.date_from]
            allocation_elveton = allocation_elveton.astype(int)

            for k,value in enumerate(allocation_elveton.iloc[0].values):
                if value != 7:
                    fcr_allocation.iloc[k, 0] = value
            fcr_allocation_by_hour = fcr_allocation.resample('1H', label='left').agg({'fcr': 'first'}).\
                reindex(pd.date_range(self.date_from, self.date_from + dt.timedelta(days=1), freq='1H')).ffill().head(-1)
            for k in range(len(fcr_allocation_by_hour)):
                if fcr_allocation_by_hour.iloc[k, 0] == 0:
                    da_allocation_temp.iloc[k, 0] = 0
            if len(da_allocation[da_allocation.da != 0]) == len(da_allocation_temp[da_allocation_temp.da != 0]):
                da_allocation = da_allocation_temp
            else:
                da_zero_rev = True

        fcr_rev = sum(fcr_allocation.values*fcr_actual.values)[0]
        if strategy_by_strategy_calculation:  # calculating revenue strategy wise
            da_allocation_without_id = self.scheduling.copy()
            for i in range(len(da_actual)):
                if id_allocation.id[i] != 0:
                    try:
                        if str(id_allocation.id[i])[-1] == 'B':
                            id_rev = id_rev - id_actual.price[i] * int(str(id_allocation.id[i])[:-1]) + da_actual.price[i+1] * int(str(da_allocation.da[i+1])[:-1])
                        else:
                            id_rev = id_rev + id_actual.price[i] * int(str(id_allocation.id[i])[:-1]) - da_actual.price[i + 1] * int(str(da_allocation.da[i + 1])[:-1])
                        da_allocation_without_id.da[i+1] = 0
                    except Exception as e:
                        print(e, '\n')
                        print('error during id_da actual revenue calculation \n')
            for i in range(len(da_allocation_without_id)):
                if da_allocation_without_id.da[i] != 0:
                    if str(da_allocation_without_id.da[i])[-1] == 'B':
                        da_rev = da_rev - da_actual.price[i] * int(str(da_allocation_without_id.da[i])[:-1])
                    else:
                        da_rev = da_rev + da_actual.price[i] * int(str(da_allocation_without_id.da[i])[:-1])
        else:  # calculating revenue market wise
            for i in range(len(da_actual)):
                if id_allocation.id[i] != 0:
                    if str(id_allocation.id[i])[-1] == 'B':
                        id_rev = id_rev - id_actual.price[i] * int(str(id_allocation.id[i])[:-1])
                    else:
                        id_rev = id_rev + id_actual.price[i] * int(str(id_allocation.id[i])[:-1])
            for i in range(len(da_actual)):
                if da_allocation.da[i] != 0:
                    if str(da_allocation.da[i])[-1] == 'B':
                        da_rev = da_rev - da_actual.price[i] * int(str(da_allocation.da[i])[:-1])
                    else:
                        da_rev = da_rev + da_actual.price[i] * int(str(da_allocation.da[i])[:-1])
        if fcr_elveton:
            if da_zero_rev:
                da_rev = 0
            else:
                # gradual reduction of da_rev based on the amount of 4-hour interval ot technical availability
                reduction_key = 0
                da_reduction_dict = {0: 0, 1: 0, 2: 0.5, 3: 0.5, 4: 1, 5: 1, 6: 1}
                for value in list(allocation_elveton.iloc[0]):
                    if value < 7:
                        reduction_key += 1
                da_rev = da_rev * (1-da_reduction_dict[reduction_key])

        return fcr_rev, da_rev, id_rev

    def get_actual_prices(self):
        # run it after 4pm cet prev day
        da_actual, fcr_actual = self.fcr_vs_da.get_actual_prices()
        id_actual = self.id_vs_da.get_actual_prices()

        actual = self.scheduling.iloc[:, 1:3]
        for i in range(len(self.scheduling)):
            if self.scheduling.da[i] != 0:
                actual.da[i] = da_actual.price[i]
            if self.scheduling.id[i] != 0:
                actual.id[i] = id_actual.price[i]

        # add type of position for each hour, so margin can be calculated correctly
        actual['type'] = None
        for i in range(len(self.scheduling)):
            if actual.da[i] == 0 and actual.id[i] == 0:
                actual.type[i] = 0
            elif actual.da[i] != 0:
                actual.type[i] = self.scheduling.da[i][-1]
            else:
                actual.type[i] = self.scheduling.id[i][-1]
        actual['xbid_price'] = None

        self.actual_prices = actual

        return actual


class backtester:
    """
    To backtest the only the auctions revenue (fcr,da,id) from current production strategy, for a given interval.
    """
    def __init__(self, date_from, date_to, fcr_elveton=False, fcr_technical_availability=False):
        self.date_from = date_from
        self.date_to = date_to
        self.revenue = []
        self.cycles = []
        self.amount_of_fault_days = 0
        self.revenue_results = pd.DataFrame(columns=['fcr', 'da', 'id'], index=pd.date_range(self.date_from, self.date_to + dt.timedelta(days=1), freq='D', closed='left'))
        self.cycle_results = pd.DataFrame(columns=['da', 'id'], index=pd.date_range(self.date_from, self.date_to + dt.timedelta(days=1), freq='D', closed='left'))
        self.fcr_elveton = fcr_elveton
        self.fcr_technical_availability = fcr_technical_availability
        if not self.fcr_elveton and self.fcr_technical_availability:
            raise Exception('fcr_technical_availability cant be True, while fcr_elveton is False\n'
                            'fcr_technical_availability is considered a special case for fcr_elveton')

    def get_auction_cycles(self, scheduling, da_zero=False):
        da_actions = 0
        id_actions = 0
        for i in range(len(scheduling)):
            if scheduling.da[i] !=0:
                da_actions = da_actions + int(scheduling.da[i][:-1])
            if scheduling.id[i] != 0:
                id_actions = id_actions + int(scheduling.id[i][:-1])
        da_cycles = da_actions/20 - id_actions/20
        id_cycles = id_actions/10

        if da_zero: # if da is zero, cause of technicalities ton elveton, count also da_cycles as zero
            da_cycles = 0
        return [da_cycles, id_cycles]

    def run(self):
        if not self.fcr_technical_availability:
            for date in pd.date_range(self.date_from, self.date_to, freq='D'):
                try:
                    strat = combined_strategy(date, date)
                    strat.get_schedule()
                    scheduling = strat.scheduling
                    self.revenue.append(strat.get_revenue_actual(fcr_elveton=self.fcr_elveton))
                    self.revenue_results.loc[date] = strat.get_revenue_actual(fcr_elveton=self.fcr_elveton)
                    if self.revenue[-1][1] == 0:
                        self.cycles.append(self.get_auction_cycles(scheduling, da_zero=True))
                        self.cycle_results.loc[date] = self.get_auction_cycles(scheduling, da_zero=True)
                    else:
                        self.cycles.append(self.get_auction_cycles(scheduling))
                        self.cycle_results.loc[date] = self.get_auction_cycles(scheduling)

                except Exception as e:
                    logging.basicConfig(filename='backtester.log', filemode='a')
                    logging.info(f'{date.strftime("%Y-%m-%d")}\n')
                    logging.exception(e)
                    self.amount_of_fault_days += 1
                    continue
        else:
            for date in pd.date_range(self.date_from, self.date_to, freq='D'):
                try:
                    strat = combined_strategy(date, date, technical_availability_constraint=True)
                    strat.get_schedule()
                    scheduling = strat.scheduling
                    self.revenue.append(strat.get_revenue_actual(fcr_elveton=self.fcr_elveton))
                    self.revenue_results.loc[date] = strat.get_revenue_actual(fcr_elveton=self.fcr_elveton)
                    if self.revenue[-1][1] == 0:
                        self.cycles.append(self.get_auction_cycles(scheduling, da_zero=True))
                        self.cycle_results.loc[date] = self.get_auction_cycles(scheduling, da_zero=True)
                    else:
                        self.cycles.append(self.get_auction_cycles(scheduling))
                        self.cycle_results.loc[date] = self.get_auction_cycles(scheduling)
                except Exception as e:
                    logging.basicConfig(filename='backtester.log', filemode='a')
                    logging.info(f'{date.strftime("%Y-%m-%d")}\n')
                    logging.exception(e)
                    self.amount_of_fault_days += 1
                    continue

        self.revenue_results = self.revenue_results.dropna()
        self.cycle_results = self.cycle_results.dropna()
        y = self.revenue_results
        c = self.cycle_results
        adjustment = ((self.date_to - self.date_from).days + 1) / len(y)
        rev = {'fcr': sum(y.fcr.values)*adjustment, 'da': sum(y.da.values)*adjustment, 'id': sum(y.id.values)*adjustment}
        cycles = {'da': np.mean(c.da.values), 'id': np.mean(c.id.values)}
        print(f'{rev}\n {cycles}')
        return [self.revenue_results, self.cycle_results]


date = dt.date(2023, 7, 2)
a = combined_strategy(date, date)
a.get_schedule()
fcr_forecast_rev, da_forecast_rev = a.get_revenue_forecast(inspection=True)
rev = a.get_revenue_actual()
x = 0
for i in range(len(a.scheduling)):
    if isinstance(a.scheduling.da[i], str):
         x += 1
if x == 4:
    cycle = 0.6
else:
    cycle = 0.3


def manual_execute():
    # a.get_actual_prices()
    # bid_proposals = a.get_bid_proposals()
    # date_from = dt.date(2023, 5, 31)
    # date_to = dt.date(2023, 6, 14)
    # x = pd.DataFrame(columns=['fcr', 'da', 'dacycles'])
    # g=0
    # for date in pd.date_range(date_from, date_to, freq='D'):
    #     a = combined_strategy(date, date)
    #     a.get_schedule()
    #     rev = a.get_revenue_actual()
    #     for i in range(len(a.scheduling)):
    #         if isinstance(a.scheduling.da[i], str):
    #             g += 1
    #     if g == 4:
    #         cycle = 0.6
    #     else:
    #         cycle = 0.3
    #     x.loc[len(x)] = [rev[0], rev[1], cycle]
    # #
    # # # b = backtester(date_from, date_to, fcr_elveton=True, fcr_technical_availability=False)
    # # # b.run()

    return None