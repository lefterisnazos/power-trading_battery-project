import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import pickle
import datetime as dt
from itertools import product
import cvxpy as cp

import wapi
from enscohelper.database import DBConnector
from enscohelper.datafeed import ActualData, ForecastData, FCRData, EnscoData, Futures
import warnings
from itertools import chain
from fcr_vs_da_production import day_ahead_production

warnings.filterwarnings('ignore')


def add_weekday_hour_feature(x):
    x['weekday'], x['hour'] = None, None
    for i in range(len(x)):
        if x.index.weekday[i] <= 4:
            x['weekday'][i] = 1
        else:
            x['weekday'][i] = 0
        x['hour'][i] = x.index.hour[i]

    return x


def consecutive_integer_check(list, number):
    for i in list:
        if np.abs(i % 23 - number % 23) == 1:
            return False
    return True


class id_vs_da_production:

    def __init__(self, date, days_back=5, margin_barrier=20, frequency='1H'):
        self.x_after_sell = None
        self.days_back = dt.timedelta(
            days=days_back)  # if days back is 2 it means we take today's values + 2 days back. So 3 days in total
        self.date_from = date - self.days_back
        self.date_to = date
        self.margin_barrier = margin_barrier
        self.frequency = frequency

    def strategy(self, date, frequency='1H'):

        date_from = date - self.days_back
        date_to = date

        hours = day_ahead_production.hours_to_trade(date, date)
        spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample(frequency,
                                                                                              label='left').mean().ffill()
        intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index(
            'datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
        intraday_auction_1h = intraday_auction.resample(frequency, label='left').agg({'price': 'mean'}).ffill()

        spot_vs_intraday = spot.join(intraday_auction_1h.rename(columns={'price': 'price15'}))
        spot_vs_intraday['15minus_spot'] = - spot_vs_intraday.price + spot_vs_intraday.price15
        x = spot_vs_intraday
        x['s_15-b_spot'] = x.price15 - x.price.shift(-1)

        buy1, buy2, sell1, sell2 = hours[0][0], hours[0][1], hours[1][0], hours[1][1]
        x_index_after_buy = ((x.index.hour > buy1) & (x.index.hour < sell1 - 1)) | (
                    (x.index.hour > buy2) & (x.index.hour < sell2 - 1))
        x_index_after_sell = ((x.index.hour > sell1) & (x.index.hour < buy2 - 1)) | (
                    (x.index.hour > sell2) | (x.index.hour < buy1 - 1))

        x_after_buy = (x.price15 - x.price.shift(-1))[x_index_after_buy]
        x_after_sell = (-x.price15 + x.price.shift(-1))[x_index_after_sell]

        x_after_buy = x_after_buy.groupby(x_after_buy.index.hour).mean()
        x_after_sell = x_after_sell.groupby(x_after_sell.index.hour).mean()
        self.x_after_buy, self.x_after_sell = x_after_buy, x_after_sell

        list_after_buy, list_after_sell = self.index_selector(x_after_sell, x_after_buy, hours)

        return list_after_buy, list_after_sell

    def index_selector(self, x_after_sell, x_after_buy, hours: list):
        buy_hours = hours[0]
        sell_hours = hours[1]

        list_after_sell, list_after_buy = [], []
        flat_hours = list(chain.from_iterable(hours))
        flat_hours.sort()
        if 0 not in flat_hours:
            flat_hours.insert(0, 0)
        if 23 not in flat_hours:
            flat_hours.append(23)
        # temp Series index isnt a datetime here, but just a plain index
        for i in range(len(flat_hours) - 1):
            if flat_hours[i] in sell_hours or flat_hours[i] == 0:
                temp = x_after_sell[(x_after_sell.index >= flat_hours[i]) & (x_after_sell.index <= flat_hours[i + 1])]
                temp = temp[temp >= self.margin_barrier].sort_values(ascending=False)
                n = len(temp)
                if temp.empty:
                    continue
                elif len(temp) == 1:
                    list_after_sell.append(temp.index[0])
                else:
                    for m in range(n - 1):
                        if m == 0:
                            list_after_sell.append(temp.index[0])
                        else:
                            if consecutive_integer_check(list_after_sell, temp.index[m]):
                                list_after_sell.append(temp.index[m])
            else:
                temp = x_after_buy[(x_after_buy.index >= flat_hours[i]) & (x_after_buy.index <= flat_hours[i + 1])]
                temp = temp[temp >= self.margin_barrier].sort_values(ascending=False)
                n = len(temp)
                if temp.empty:
                    continue
                elif len(temp) == 1:
                    list_after_buy.append(temp.index[0])
                else:
                    for m in range(n - 1):
                        if m == 0:
                            list_after_buy.append(temp.index[0])
                        else:
                            if consecutive_integer_check(list_after_buy, temp.index[m]):
                                list_after_buy.append(temp.index[m])

        return list_after_buy, list_after_sell

    def strategy_tester(self, date_from, date_to):
        spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample(self.frequency,
                                                                                              label='left').mean().ffill()
        intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index(
            'datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
        intraday_auction_1h = intraday_auction.resample(self.frequency, label='left').agg({'price': 'mean'}).ffill()

        spot_vs_intraday = spot.join(intraday_auction_1h.rename(columns={'price': 'price15'}))
        spot_vs_intraday['15minus_spot'] = - spot_vs_intraday.price + spot_vs_intraday.price15
        x = spot_vs_intraday
        x_after_buy = (x.price15 - x.price.shift(-1))
        x_after_sell = (-x.price15 + x.price.shift(-1))

        results = pd.DataFrame(columns=['spread', 'type', 'datetime'])
        for date in pd.date_range(date_from, date_to - dt.timedelta(days=1), freq='D'):
            index_after_buy, index_after_sell = self.strategy(date)
            test_date = date + dt.timedelta(days=1)
            test_after_buy = x_after_buy[(x_after_buy.index.date == test_date)]
            test_after_sell = x_after_sell[(x_after_sell.index.date == test_date)]
            for j in range(len(test_after_buy)):
                if test_after_buy.index.hour[j] in index_after_buy:
                    results.loc[len(results), 'spread'] = test_after_buy[j]
                    results.loc[len(results) - 1, 'type'] = 'after_buy'
                    results.loc[len(results) - 1, 'datetime'] = test_after_buy.index[j]
                elif test_after_sell.index.hour[j] in index_after_sell:
                    results.loc[len(results), 'spread'] = test_after_sell[j]
                    results.loc[len(results) - 1, 'type'] = 'after_sell'
                    results.loc[len(results) - 1, 'datetime'] = test_after_sell.index[j]
                else:
                    continue

        avg_margin = (np.mean(results.spread))
        total = avg_margin * len(results)
        results_info = [avg_margin, total, len(results)]
        return results_info

    def get_allocation_results(self, quantity='x'):
        date = self.date_to
        list_after_buy, list_after_sell = self.strategy(date)
        df = pd.DataFrame(columns=['da', 'id'], index=pd.date_range(date, date + dt.timedelta(days=1), freq='1H', closed='left'))
        for i in range(len(df)-1):
            if df.index[i].hour in list_after_buy:
                df.id[i] = f'{quantity}S'
                df.da[i+1] = f'{quantity}B'
            elif df.index[i].hour in list_after_sell:
                df.id[i] = f'{quantity}B'
                df.da[i+1] = f'{quantity}S'

        return df

    def get_revenue_forecast(self, quantity=1):
        date = self.date_to
        list_after_buy, list_after_sell = self.strategy(date)
        df = pd.DataFrame(columns=['id_da'], index=pd.date_range(date, date + dt.timedelta(days=1), freq='1H', closed='left'))
        for i in range(len(df)):
            if df.index[i].hour in list_after_buy:
                df.id_da[i] = self.x_after_buy[i]*quantity
            elif df.index[i].hour in list_after_sell:
                df.id_da[i] = self.x_after_sell[i]*quantity

        return df

    def get_bid_proposals(self):
        date = self.date_to
        days_back = dt.timedelta(days=600)
        da_bid = day_ahead_production(self.date_to, self.date_to).get_bid_proposals()
        id_bid = da_bid * 1.2
        intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date, date, area=str(23)).set_index('datetime').sort_index().drop(
            columns=['datetime_id', 'area_id'])
        id_bid = id_bid.resample('15min', closed='left').agg(['last']).ffill()
        id_bid = id_bid.reindex(intraday_auction.index, method='ffill')

        return id_bid

    def get_actual_prices(self, frequency='1H'):
        date_from = self.date_to
        date_to = self.date_to
        intraday_auction_1h = None
        try:
            intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index('datetime').sort_index().drop(
                columns=['datetime_id', 'area_id'])
            intraday_auction_1h = intraday_auction.resample(frequency, label='left').agg({'price': 'mean'}).ffill()
        except:
            print(f'error loading ID actualdata for {self.date_to.strftime("%Y%m%d%H%M")}')

        return intraday_auction_1h


def backtest_strat_params(date_from=dt.date(2021, 9, 15), date_to=dt.date(2022, 11, 27), days_back=30, margin_barrier=45):
    backtest = pd.DataFrame(columns=['days_back', 'margin_barrier', 'avg_margin', 'total', 'actions', 'avg_margin/action'])
    for i in range(20, days_back + 1, 5):
        for j in range(5, margin_barrier + 1, 5):
            x = id_vs_da_production(dt.date.today(), days_back=i, margin_barrier=j)
            result = x.strategy_tester(date_from, date_to)
            result = [i, j] + result + [result[0]/result[2]]
            backtest.loc[len(backtest)] = result
            del x

    return backtest

# a = id_vs_da_production(dt.date.today())
# b = backtest_strat_params(dt.date(2022, 1, 1), dt.date(2022, 7, 1))