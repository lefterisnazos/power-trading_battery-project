import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import pickle
import datetime as dt
from itertools import product
import wapi
from enscohelper import date_time
from enscohelper.database import DBConnector
from enscohelper.datafeed import ActualData, ForecastData, FCRData, EnscoData
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from battery_simulator import Battery, Position
import math
import re


class day_ahead_production:

    def __init__(self, date_from, date_to, scheduling=True, da_defensiveness=0.499):

        self.date_from = date_from
        self.date_to = date_to
        self.da_defensiveness = da_defensiveness
        if scheduling is True:
            self.forecast_spot = ForecastData().get_spot_prices('Germany', date_from, date_to)
        else:
            self.actual_spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to)

    @staticmethod
    def hours_to_trade(date_from, date_to):
        forecast_spot = ForecastData().get_spot_prices('Germany', date_from, date_to)
        first_12hour = forecast_spot[forecast_spot.index.hour <= 11]
        second_12hour = forecast_spot[forecast_spot.index.hour > 11]
        first_buy_sell = day_ahead_production.find_max_spread(first_12hour)
        second_buy_sell = day_ahead_production.find_max_spread(second_12hour)
        if None in first_buy_sell or None in second_buy_sell or 0 in first_buy_sell or 0 in second_buy_sell:
            sell_hours = [first_12hour.idxmax()[0].hour, second_12hour.idxmax()[0].hour]
            buy_hours = [first_12hour.idxmin()[0].hour, forecast_spot[(forecast_spot.index.hour >= sell_hours[0]) & (forecast_spot.index.hour <= sell_hours[1])].idxmin()[0].hour]
            return buy_hours, sell_hours
        else:
            return [first_buy_sell[0].hour, second_buy_sell[0].hour], [first_buy_sell[1].hour, second_buy_sell[1].hour]

    def get_results(self):
        day_ahead = self.forecast_spot.copy()
        day_ahead['type'] = None
        buy_hours, sell_hours = day_ahead_production.hours_to_trade(self.date_from, self.date_to)
        for index in day_ahead.index:
            if index.hour in sell_hours:
                day_ahead['type'][day_ahead.index == index] = 'S'
            elif index.hour in buy_hours:
                day_ahead['type'][day_ahead.index == index] = 'B'
        day_ahead = day_ahead.dropna()
        day_ahead_spread = day_ahead.price.diff()
        day_ahead_spread.drop([day_ahead_spread.index[0], day_ahead_spread.index[2]], inplace=True)
        day_ahead_spread = day_ahead_spread.groupby(pd.Grouper(freq='12H')).sum()
        day_ahead_spread = day_ahead_spread.to_frame().rename(columns={0: 'price'})
        day_ahead = self.da_allocation_optimizer(day_ahead, day_ahead_spread)

        return day_ahead, day_ahead_spread, [buy_hours, sell_hours]

    def get_bid_proposals(self):
        date_from = self.date_from
        date_to = self.date_to

        forecast = ForecastData().get_spot_prices('Germany', date_from, date_to)
        df = forecast
        df['daily_mean'] = df.price.resample('D').agg({'price': 'mean'})
        df['daily_mean'] = df['daily_mean'].ffill()
        df['bid_proposal'] = None
        for i in range(len(df)):
            if df.price[i] > 100:
                df.bid_proposal[i] = df.price[i] + np.abs(df.daily_mean[i])
            elif df.price[i] <= 100:
                if df.price[i] > 0:
                    df.bid_proposal[i] = (df.price[i] + np.abs(df.daily_mean[i])) * 1.35
                else:
                    if df.daily_mean[i] >= 10:
                        df.bid_proposal[i] = (df.price[i] + np.abs(df.daily_mean[i] * 2))
                    else:
                        df.bid_proposal[i] = (df.price[i] / 3 + 10)

        return df[['bid_proposal']]

    def da_allocation_optimizer(self, da_trades, da_spread):
        da_spread['rev_fraction'] = None
        # the higher the parameter the more defensive the strategy. max value 0.5
        c = self.da_defensiveness
        if len(da_trades) == 4:
            for i in range(len(da_spread)):
                da_spread['rev_fraction'][i] = da_spread.iloc[i, 0] / np.sum(da_spread.iloc[:, 0].values)
            if da_trades.price[3] >= da_trades.price[1]:
                if da_spread.iloc[1, 1] < c:
                    return da_trades.drop(index=(da_trades.index[1])).drop(index=(da_trades.index[2]))
                elif da_spread.iloc[0, 1] < c:
                    return da_trades.iloc[2:, :]
                else:
                    return da_trades
            else:
                if da_spread.iloc[1, 1] < c:
                    return da_trades.iloc[:2, :]
                elif da_spread.iloc[0, 1] < c:
                    return da_trades.iloc[2:, :]
                else:
                    return da_trades
        else:
            sells = da_trades[da_trades.type == 'S']
            buys = da_trades[da_trades.type == 'B']
            if len(sells) == 2:
                sells = sells[sells.price == sells.price.max()]
            if len(buys) == 2:
                buys = buys[buys.price == buys.price.max()]
            da_trades = pd.concat([buys, sells])
            da_trades.sort_index()
            return da_trades

    def bid_proposals_backtesting(self):
        date = self.date_from
        days_back = dt.timedelta(days=600)

        forecast = ForecastData().get_spot_prices('Germany', date - days_back, date)
        actual = ActualData().get_spot_prices('Germany_Luxemburg', date - days_back, date)
        intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date - days_back, date, area=str(23)).set_index('datetime').sort_index().drop(
            columns=['datetime_id', 'area_id'])
        intraday_auction_1h = intraday_auction.resample('1H', label='left').agg({'price': 'mean'}).ffill()
        percentage_error = (actual - forecast) / forecast
        nominal_error = (actual - forecast)

        merged = actual.join(forecast.rename(columns={'price': 'forecast_price'}))
        merged = merged.join(nominal_error.rename(columns={'price': 'nominal_error'})).join(percentage_error.rename(columns={'price': 'percentage_error'}))
        merged['daily_mean'] = merged.forecast_price.resample('D').agg({'price': 'mean'})
        merged['daily_mean'] = merged['daily_mean'].ffill()
        merged['bid_proposal'] = None
        merged['id'] = intraday_auction_1h
        for i in range(len(merged)):
            if merged.forecast_price[i] > 100:
                merged.bid_proposal[i] = merged.forecast_price[i] + np.abs(merged.daily_mean[i])
            elif merged.forecast_price[i] <= 100:
                if merged.forecast_price[i] > 0:
                    merged.bid_proposal[i] = (merged.forecast_price[i] + np.abs(merged.daily_mean[i])) * 1.35
                else:
                    if merged.daily_mean[i] >= 10:
                        merged.bid_proposal[i] = (merged.forecast_price[i] + np.abs(merged.daily_mean[i] * 2))
                    else:
                        merged.bid_proposal[i] = (merged.forecast_price[i] / 3 + 10)

        return merged[['bid_proposal']]

    def get_actual_prices(self):
        actual_da = ActualData().get_spot_prices('Germany_Luxemburg', self.date_from, self.date_to).resample('1H', label='left').mean().ffill()

        return actual_da

    def get_forecast_prices(self):
        forecast_da = ForecastData().get_spot_prices('Germany', self.date_from, self.date_to).resample('1H', label='left').mean().ffill()
        return forecast_da

    @staticmethod
    def find_max_spread(df):
        max_spread = 0
        index_low = None
        index_high = None
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                spread = df.iloc[j]['price'] - df.iloc[i]['price']
                if spread > max_spread:
                    max_spread = spread
                    index_low = df.index[i]
                    index_high = df.index[j]
        return index_low, index_high, max_spread


class fcr_production:

    def __init__(self, date_from, date_to, scheduling=True):

        self.date_from = date_from
        self.date_to = date_to
        self.scheduling = scheduling
        self.fcr = self.get_data(date_from, date_to, scheduling=False)

    def replace_hours(self, index: dt.datetime, product_name):
        return index.replace(hour=int(product_name[-5:-3]))

    def get_data(self, date_from, date_to, scheduling=True):
        if scheduling is False:
            fcr = FCRData().get_auction_results(date_from, date_to)
            fcr = fcr[(fcr['area'] == 'DE') & (fcr['tender_number'] == 1)]
            fcr['date'] = pd.to_datetime(fcr['date'])
            fcr.index = map(self.replace_hours, fcr.date, fcr.product_name)
            return fcr
        else:
            fcr = FCRData().get_fcr_price_forecast(date_from, date_to).set_index('datetime').drop(columns=['area_id']).resample('4H', label='left', origin='end_day').\
                    agg({'price': 'sum'}).ffill()
            fcr = fcr[(fcr.index.date >= self.date_from) & (fcr.index.date <= self.date_to)]

            return fcr

    def get_results(self, frequency='12H'):
        scheduling = self.scheduling
        if scheduling is False:
            fcr = self.fcr.copy()
            fcr = fcr[['settlement_capacity_price']]
            fcr.rename(columns={'settlement_capacity_price': 'eur/mwh'}, inplace=True)
            fcr_frequency = fcr.groupby(pd.Grouper(freq=frequency)).sum()

            return [fcr, fcr_frequency]
        else:
            fcr = self.fcr.copy()
            fcr.rename(columns={'price': 'eur/mwh'}, inplace=True)
            fcr_frequency = fcr.groupby(pd.Grouper(freq=frequency)).sum()

            return [fcr, fcr_frequency]

    def get_bid_proposals(self):
        fcr_forecast = self.get_data(self.date_from, self.date_to)
        fcr_forecast = fcr_forecast.rename(columns={'price': 'forecast_price'})
        fcr_forecast['bid_proposal'] = fcr_forecast.iloc[:, [0]]/3

        # fcr = FCRData().get_auction_results(self.date_from, self.date_to)
        # fcr = fcr[(fcr['area'] == 'DE') & (fcr['tender_number'] == 1)]
        # fcr['date'] = pd.to_datetime(fcr['date'])
        # fcr.index = map(self.replace_hours, fcr.date, fcr.product_name)
        # fcr.sort_index(ascending=True, inplace=True)
        # fcr_actual = fcr[['settlement_capacity_price']].join(fcr_forecast)

        return fcr_forecast[['bid_proposal']].rename(columns={'bid_proposal': 'fcr_bid'})

    def get_actual_prices(self):
        fcr = FCRData().get_auction_results(self.date_from, self.date_to)
        fcr = fcr[(fcr['area'] == 'DE') & (fcr['tender_number'] == 1)]
        fcr['date'] = pd.to_datetime(fcr['date'])
        fcr.index = map(self.replace_hours, fcr.date, fcr.product_name)
        fcr = fcr[['settlement_capacity_price']]
        fcr.rename(columns={'settlement_capacity_price': 'eur/mwh'}, inplace=True)
        fcr = fcr.groupby(pd.Grouper(freq='4H')).sum()

        return fcr

    def excel_format_results(self, date_from, date_to):
        results = self.get_data(date_from, date_to, scheduling=False)
        results = results[['settlement_capacity_price']]
        results = results.sort_index(ascending=True)
        index = pd.date_range(dt.date(2022, 1, 1), dt.date(2022, 12, 31), freq='D')
        df = pd.DataFrame(columns=['0-4', '4-8', '8-12', '12-16', '16-20', '20-24'], index=index)
        k = 0
        for i in range(len(df)):
            for j in range(6):
                df.iloc[i, j] = results.iloc[k, 0]
                k += 1
        return df


class fcr_vs_da_production:

    def __init__(self, date):
        self.forecast_spot = None
        self.date = date
        self.da = day_ahead_production(date, date).get_results()
        self.fcr = fcr_production(date, date, scheduling=True).get_results()
        self.fcr_quantity = 7
        self.da_quantity = 3

    def get_allocation_results(self):
        df = pd.DataFrame(columns=['fcr', 'da'], index=pd.date_range(self.date, self.date + dt.timedelta(days=1), freq='1H', closed='left'))

        da_spread = self.da[1].rename(columns={'price': 'da_eur_mwh'})
        fcr = self.fcr[1].rename(columns={'eur/mwh': 'fcr_eur_mwh'})
        merged = fcr.iloc[:, [2]].join(da_spread)
        merged['win'] = None

        for i in range(len(merged)):
            if merged[['da_eur_mwh']].iloc[i].isnull()[0]:
                merged.iloc[i, 1] = 0  # replace empty values, of da_spread with 0.
        for i in range(len(merged)):
            if merged.iloc[i, 0] > merged.iloc[i, 1]:
                merged.win[i] = 'fcr'
            else:
                merged.win[i] = 'da'

        da_strat = self.da[0]
        da_strat_index = da_strat.index
        da_strat = da_strat.resample('1H').asfreq()
        index = pd.DatetimeIndex(pd.date_range(self.date, self.date + dt.timedelta(days=1), freq='1H', closed='left'))
        da_strat = da_strat.reindex(index)

        for i in range(len(df)):
            if df.index[i].hour < 12:
                if merged.win[0] == 'fcr':
                    df['fcr'][i] = self.fcr_quantity
                    if df.index[i].hour in da_strat_index.hour:
                        df['da'][i] = f'{self.da_quantity}{da_strat.type[i]}'
                else:
                    df['fcr'][i] = self.da_quantity
                    if df.index[i].hour in da_strat_index.hour:
                        df['da'][i] = f'{self.fcr_quantity}{da_strat.type[i]}'
            else:
                if merged.win[1] == 'fcr':
                    df['fcr'][i] = self.fcr_quantity
                    if df.index[i].hour in da_strat_index.hour:
                        df['da'][i] = f'{3}{da_strat.type[i]}'
                else:
                    df['fcr'][i] = self.da_quantity
                    if df.index[i].hour in da_strat_index.hour:
                        df['da'][i] = f'{self.fcr_quantity}{da_strat.type[i]}'

        df = self.allocation_results_ERROR_correction(df, da_strat[['type']])

        # here we are removing a da spread if its very small
        if len(da_strat[(da_strat.type == 'S') | (da_strat.type == 'B')]) == 4:
            da_spread['revenue_fraction'] = None
            for i in range(len(da_spread)):
                da_spread['revenue_fraction'][i] = da_spread.iloc[i, 0]/np.sum(da_spread.iloc[:, 0].values)
                if da_spread['revenue_fraction'][i] < 0.2:
                    no_da_index = (df.index >= da_spread.index[i]) & (df.index < da_spread.index[i] + dt.timedelta(hours=12))
                    df.fcr[no_da_index] = self.fcr_quantity
                    df.da[no_da_index] = np.nan

        return df

    def allocation_results_ERROR_correction (self, df, da_strat_type):
        # basically if there is an error in the quantities of  how they add up, for example if a 7B 3S occurs, we make it 3B 3S
        cum_da_quantity = 0  # this should be 0 at end of scheduling
        for allocation in df.da:
            if not isinstance(allocation, str):
                continue
            else:
                if allocation[-1] == 'S':
                    cum_da_quantity = cum_da_quantity - int(str(allocation[:-1]))
                else:
                    cum_da_quantity = cum_da_quantity + int(str(allocation[:-1]))
        if cum_da_quantity != 0:
            for i in range(len(df)):
                df.iloc[i, 0] = 7
                if isinstance(df.iloc[i, 1], str):
                    df.iloc[i, 1] = str(3) + da_strat_type.type.iloc[i]
        else:
            pass

        return df

    def get_revenue_forecast(self):
        df = self.get_allocation_results()
        fcr_allocation = df['fcr'].resample('4H').agg('first')
        fcr_forecast_data = fcr_production(self.date, self.date, scheduling=True).get_data(self.date, self.date, scheduling=True)
        fcr_rev = fcr_forecast_data.iloc[:, 0].mul(fcr_allocation).to_frame().rename(columns={0:'revenue'})

        da_rev = self.da[0]
        da_rev['quantity'], da_rev['revenue'] = None, None
        k=0
        for i in da_rev.index:
            temp = df[df.index == i].iloc[0, 1]
            da_rev['quantity'][k] = int(temp[:-1])
            k = k+1
        for i in range(len(da_rev)):
            if da_rev.type[i] == 'B':
                da_rev['revenue'][i] = - da_rev.price[i] * da_rev.quantity[i]
            else:
                da_rev['revenue'][i] = da_rev.price[i] * da_rev.quantity[i]

        return fcr_rev, da_rev

    def get_bid_proposals(self):
        date = self.date
        da_bid = day_ahead_production(date, date).get_bid_proposals()
        fcr_bid = fcr_production(date, date, scheduling=True).get_bid_proposals()

        return da_bid, fcr_bid

    def get_actual_prices(self):
        date = self.date
        da_actual = day_ahead_production(date, date).get_actual_prices()
        fcr_actual = fcr_production(date, date).get_actual_prices()

        return da_actual, fcr_actual

# date = dt.date.today() - dt.timedelta(days=4)
# a = day_ahead_production(dt.date(2022, 9, 14), dt.date(2022,12,2))
# a.get_bid_proposals()
# x=2
