import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import pickle
import datetime as dt
from itertools import product

import pmdarima
import wapi
from enscohelper.database import DBConnector
from enscohelper.datafeed import ActualData, ForecastData, FCRData, EnscoData, Futures
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from battery_simulator import Battery, Position
from scipy.signal import argrelextrema
import time
import pprint
import math
import openpyxl
from fcv_vs_da import fcr_strat, day_ahead_strat

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')


def consecutive_duplicates_trimmer(df):
    df = pd.Series(df)
    for i in range(len(df) - 1):
        if df[i] is not None and df[i] == df[i + 1]:
            df[i + 1] = None
            k = i + 1
            flag = True
            while flag is True:
                if df[k + 1] == df[i]:
                    df[k + 1] = None
                    k = k + 1
                else:
                    i = k
                    flag = False
    return df


def add_weekday_hour_feature(x):

    x['weekday'], x['hour'] = None, None
    for i in range(len(x)):
        if x.index.weekday[i] <= 4:
            x['weekday'][i] = 1
        else:
            x['weekday'][i] = 0
        x['hour'][i] = x.index.hour[i]

    return x


class intraday_auction_15:

    def __init__(self, date_from, months_back=12, frequency='1H'):
        self.intraday_auction = None
        self.spot_forecast, self.spot_actual = None, None
        self.fundamentals_actual_forecast_15, self.fundamentals_forecast_15 = None, None

        self.battery = Battery(date_from)
        self.position = Position()
        self.months = months_back
        self.drop_columns = ['consumption', 'temperature', 'datetime_id']
        self.frequency = frequency

    def create_features_labels(self, date_from, date_to, frequency='1H'):
        forecast_data = ForecastData()
        actual_data = ActualData()
        ensco_data = EnscoData()
        fcr_data = FCRData()
        area = 'Germany'
        area_number = 23

        valley = ensco_data.valley_germany(date_from, date_to)
        valley_index = valley[valley.prediction == True].index

        self.fundamentals_actual_forecast_15 = fcr_data.get_intraday_fundamentals_15min_actual_forecast(date_from, date_to, area=str(area_number)).\
            set_index('datetime').drop(columns=self.drop_columns).resample(self.frequency,  label='left').mean().ffill()

        gas = Futures().get_gas_futures(date_from, date_to + dt.timedelta(days=1)).set_index('trade_date').fillna(method='bfill')
        gas.index = pd.DatetimeIndex(gas.index)
        gas = gas['settlement_price']
        gas = gas.groupby(gas.index.date).mean()
        gas.index = pd.DatetimeIndex(gas.index)
        gas = gas.resample(self.frequency, label='left').agg({'price': 'last'}).ffill()
        gas = gas.iloc[:-1, :].rename(columns={'price': 'gas'})

        self.spot_actual = actual_data.get_spot_prices("Germany_Luxemburg", date_from, date_to).resample('D',  label='left').agg({'price': 'last'}).\
            ffill().reindex(self.fundamentals_actual_forecast_15.index, method='ffill')

        self.intraday_auction = fcr_data.get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(area_number)).set_index('datetime').\
            sort_index().drop(columns=['datetime_id', 'area_id']).resample(self.frequency,  label='left').agg({'price': 'mean'}).ffill()

        x = self.fundamentals_actual_forecast_15.join(gas)
        x = add_weekday_hour_feature(x)

        y = self.intraday_auction.copy()

        return [x, y]

    def intraday_auction_predict(self, date_from, date_to):

        forecast_data = ForecastData()
        fcr_data = FCRData()
        area_number = 23

        x, y = self.create_features_labels(date_from - pd.DateOffset(months=self.months), date_to + dt.timedelta(days=1))

        # self.fundamentals_forecast_15 = fcr_data.get_intraday_fundamentals_15min_forecast(date_from, date_to,area=str(area_number)).set_index( 'datetime').drop(columns=['datetime_id', 'area_id', 'consumption', 'temperature'])
        # self.spot_forecast = forecast_data.get_spot_prices("Germany", date_from, date_to).resample('15min', label='left', origin='end_day').agg( {'price': 'last'}).ffill().reindex(self.fundamentals_forecast_15.index, method='ffill')
        # x_forecast = self.fundamentals_forecast_15.join(self.spot_forecast)

        self.fundamentals_forecast_15 = fcr_data.get_intraday_fundamentals_15min_forecast(date_from, date_to, area=str(area_number)).\
            set_index('datetime').drop(columns= self.drop_columns).resample(self.frequency,  label='left').mean().ffill()
        gas = Futures().get_gas_futures(date_from, date_to + dt.timedelta(days=1)).set_index('trade_date').shift(1).fillna(method='bfill')
        gas.index = pd.DatetimeIndex(gas.index)
        gas = gas['settlement_price']
        gas = gas.groupby(gas.index.date).mean()
        gas.index = pd.DatetimeIndex(gas.index)
        gas = gas.resample(self.frequency, label='left').agg({'price': 'last'}).ffill()
        gas = gas.iloc[:-1, :].rename(columns={'price': 'gas'})
        x_forecast = self.fundamentals_forecast_15.join(gas)
        x_forecast = add_weekday_hour_feature(x_forecast)

        predictions = None
        clf = None
        train_score = []
        mape = []
        r2 = []

        start = time.time()
        for timestamp in (pd.date_range(date_from, date_to, freq='1D')):

            x_train = x.loc[timestamp - pd.DateOffset(months=self.months):timestamp, :][:-1]
            y_train = y.loc[timestamp - pd.DateOffset(months=self.months):timestamp][:-1]

            x_test = x_forecast[(x_forecast.index >= timestamp) & (x_forecast.index < timestamp + dt.timedelta(days=1))]
            y_test = self.intraday_auction[self.intraday_auction.index.isin(x_test.index)]

            parameters = {'random_state': [0], 'n_estimators': [50, 100, 200], 'max_depth': [16]}
            parameters = {'C': [0.1], 'kernel': ['linear']}
            if clf is None or (timestamp.day % 7 == 0):
                sample_weights = np.ones(shape=(len(y_train),))
                sample_weights[int(len(sample_weights) * 0.8):] = 1.8
                clf = GridSearchCV(RandomForestRegressor(), param_grid=parameters, n_jobs=int(multiprocessing.cpu_count() * 0.8))
                #clf = SVR(C=0.1, kernel='linear')
                clf.fit(x_train, y_train, sample_weight=sample_weights)
                train_score.append(clf.score(x_train, y_train))

                #arima part

            try:
                temp = pd.DataFrame(clf.predict(x_test), index=x_test.index)

                # freq = 24
                # train, test = y_train.price[:-freq], y_test.price[-freq:]
                # model_ar = pmdarima.arima.ARIMA((3, 1, 0), (1, 0, 2, 24))
                # model_ar.fit(train)
                # pred_ar = model_ar.predict(freq)
                # pred_ar.index = pred_ar.index + dt.timedelta(days=1)
                # temp_arima = pd.DataFrame(pred_ar, index=y_test.index)
                # model_ar.update(y_test)
                # plt.clf()
                # plt.plot(y_test, c='tab:orange')
                # plt.plot(temp_arima.iloc[:,0])
                # plt.show()
            except:
                pass
            if predictions is None:
                predictions = temp
            else:
                predictions = predictions.append(temp)
                plt.clf()
                plt.plot(y_test, c='tab:orange')
                plt.plot(temp)
                plt.show()

        y_test_all_period = self.intraday_auction[self.intraday_auction.index.isin(predictions.index)]
        mape.append(mean_absolute_percentage_error(y_test_all_period, predictions))
        r2.append(r2_score(y_test_all_period, predictions))

        end = time.time()

        return [predictions.rename(columns={0:'price'}), end - start, train_score, mape, r2]

    def intraday_auction_strategy(self, date_from, date_to):

        self.intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index('datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
        self.fundamentals_actual_forecast_15 = FCRData().get_intraday_fundamentals_15min_actual_forecast(date_from, date_to, area=str(23)).set_index('datetime').drop(columns=['consumption', 'temperature'])
        self.spot_actual = ActualData().get_spot_prices("Germany_Luxemburg", date_from, date_to).resample('15min', label='left').agg({'price': 'last'}).ffill().reindex(self.fundamentals_actual_forecast_15.index, method='ffill')
        prediction, time_to_predict, train_score, mape, r2 = self.intraday_auction_predict(date_from, date_to)

        intraday_auction = self.intraday_auction.copy()
        #prediction = pd.read_excel('prediction.xlsx').set_index('datetime')
        intraday_auction['price_mw15min'] = intraday_auction.price
        prediction['price_mw15min'] = prediction.price

        n = 2 # d
        df_real = intraday_auction[intraday_auction.index.isin(prediction.index)]
        df_real['min'] = df_real.iloc[argrelextrema(df_real.price_mw15min.values, np.less_equal, order=n)[0]]['price_mw15min']
        df_real['max'] = df_real.iloc[argrelextrema(df_real.price_mw15min.values, np.greater_equal, order=n)[0]]['price_mw15min']

        df_pred = prediction[['price_mw15min']]
        df_pred.drop_duplicates(keep='first', inplace=True)
        df_pred['min'] = df_pred.iloc[argrelextrema(df_pred.price_mw15min.values, np.less_equal, order=n)[0]]['price_mw15min']
        df_pred['max'] = df_pred.iloc[argrelextrema(df_pred.price_mw15min.values, np.greater_equal, order=n)[0]]['price_mw15min']
        df_pred['min_clean'] = consecutive_duplicates_trimmer(df_pred['min'])
        df_pred['max_clean'] = consecutive_duplicates_trimmer(df_pred['max'])

        quantity = 1/4
        df_real['signal'] = None
        df_real.drop(columns=['price']).rename(columns={'price_mw_15min': 'price'})
        df_real.replace({math.nan: None}, inplace=True)
        df_pred.replace({math.nan: None}, inplace=True)

        b = (np.diff(np.sign(np.diff(df_pred.price_mw15min))) > 0).nonzero()[0] + 1  # local min
        c = (np.diff(np.sign(np.diff(df_pred.price_mw15min))) < 0).nonzero()[0] + 1

        df_pred['buy-sell'] = None
        k = 0
        for i in range(len(df_pred)):
            if df_pred.index.get_loc(df_pred.index[i]) == int(b[k]):
                df_pred['buy-sell'][i] = -1
                k = k+1
            if b[-1] <= b[k]:
                break
        k = 0
        for i in range(len(df_pred)):
            if df_pred.index.get_loc(df_pred.index[i]) == int(c[k]):
                df_pred['buy-sell'][i] = 1
                k = k+1
            if c[-1] <= c[k]:
                break

        # for i in range(len(df_pred)):
        #     if df_pred['min'][i] is not None and df_pred['max'][i] is None:
        #         self.position.add_position(df_real.price[i], quantity)
        #         self.position.save_trade(df_real.index[i], quantity, df_real.price[i], 'B')
        #         df_real['signal'][i] = 1
        #
        #         self.battery.current_total_capacity(quantity, df_real.index[i].date())
        #     elif df_pred['min'][i] is None and df_pred['max'][i] is not None:
        #         self.position.reduce_position(df_real.price[i], quantity)
        #         self.position.save_trade(df_real.index[i], quantity, df_real.price[i], 'S')
        #         self.position.add_pnl_from_trade(df_real.index[i], df_real.price[i], quantity)
        #         df_real['signal'][i] = -1
        #
        #         self.battery.current_total_capacity(quantity, df_real.index[i].date())
        #     else:
        #         continue
        for i in range(len(df_pred)):
            if df_pred['buy-sell'][i] == 1:
                # if update_soc is True, we are definitely placing a buy, so it also updates soc state. no need to add it like in the sell occasion
                if self.battery.update_soc(quantity, 'B'):
                    self.position.add_position(df_real.price[i], quantity)
                    self.position.save_trade(df_real.index[i], quantity, df_real.price[i], 'B')
                    df_real['signal'][i] = 1
            elif df_pred['buy-sell'][i] == -1:
                if self.battery.get_soc() == quantity:
                    all_quantity = quantity
                elif self.battery.get_soc() > quantity:
                    all_quantity = self.battery.get_soc()
                else:
                    print('çant have negative quantity')
                    continue
                self.position.reduce_position(df_real.price[i], all_quantity)
                self.position.save_trade(df_real.index[i], all_quantity, df_real.price[i], 'S')
                self.position.add_pnl_from_trade(df_real.index[i], df_real.price[i], all_quantity)
                self.battery.update_soc(all_quantity, 'S')
                df_real['signal'][i] = -1

            self.battery.current_total_capacity(quantity, df_real.index[i].date())

        i = 13
        plt.plot(prediction.iloc[96*i:96*(i+1), 0])
        plt.plot(df_real.iloc[96 * i:96 * (i + 1), 0])
        print(self.position.analytics()[0], '\ntotal revenue: ', self.position.analytics()[1])
        print(f'\ncycles used: {self.battery.cycles_used}, with current max_capacity: {self.battery.current_max_capacity}')


def intraday_dayahead_optimization(date_from, date_to, frequency='1H'):

    """
    buy 9 - 10 id_15min, sell 10-11 DA
    buy 11-12 id_15min, sell 12-13 DA


    buy 21-22 id_15min, sell 22-23 DA
    buy 23-24 id_15min, sell 0-1 DA
                OR
    buy 20-21 id_15min, sell 21-22 DA
    buy 22-23 id_15min, sell 23-24 DA


    sell 4-5 id_15min, buy 5-6 DA
                or
    sell 5-6 id_15min, buy 6-7 DA

    sell 15-16 id_15min, buy 16-17 DA
    sell 17-18 id_15min, buy 18-19 DA

    """
    #hours = day_ahead_strat.hours_to_trade()  # variables for after buy sell indeces

    spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample(frequency, label='left').mean().ffill()
    intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index('datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
    intraday_auction_1h = intraday_auction.resample(frequency, label='left').agg({'price': 'mean'}).ffill()

    spot_vs_intraday = spot.join(intraday_auction_1h.rename(columns={'price': 'price15'}))
    spot_vs_intraday['15minus_spot'] = - spot_vs_intraday.price + spot_vs_intraday.price15
    x = spot_vs_intraday#[(spot_vs_intraday.index.year == 2022) & (spot_vs_intraday.index.month == 10)]
    x['s_15-b_spot'] = x.price15 - x.price.shift(-1)

    x_index_after_buy = (x.index.hour > 3) & (x.index.hour < 7) | ((x.index.hour > 14) & (x.index.hour < 18))
    x_index_after_sell = (x.index.hour > 8) & (x.index.hour < 13) | ((x.index.hour > 19) | (x.index.hour < 2))

    # the following series contain the margin from sellin in ID and buying the following hour in spot, and vise versa
    # after_buy is after buying dayahead quantity, and corresponds to successive sell-buys
    # after sell is after selling dayahead quantity, and corresponds to successive buy-sells

    # make a trade first in the intraday then in the day ahead
    x_after_buy = (x.price15 - x.price.shift(-1))[x_index_after_buy]
    x_after_sell = (-x.price15 + x.price.shift(-1))[x_index_after_sell]

    #x_after_buy = (- x.price15.shift(-1) + x.price)[x_index_after_buy]
    #x_after_sell = (x.price15.shift(-1) - x.price)[x_index_after_sell]

    after_sell = x_after_sell.groupby(x_after_sell.index.hour).mean()
    after_buy = x_after_buy.groupby(x_after_buy.index.hour).mean()

    after_sell_hours = [[9, 11, 21, 23], [9, 11, 20, 22]]
    after_buy_hours = [4, 15, 17]



    from pmdarima import auto_arima


    clf=RandomForestRegressor()


    x = 2


def intraday_15_exploration(date_from, date_to, frequency='1H'):
      # variables for after buy sell indeces

    spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample(frequency,
                                                                                          label='left').mean().ffill()
    intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index(
        'datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
    intraday_auction_1h = intraday_auction.resample(frequency, label='left').agg({'price': 'mean'}).ffill()

    spot_vs_intraday = spot.join(intraday_auction_1h.rename(columns={'price': 'price15'}))
    spot_vs_intraday['15minus_spot'] = - spot_vs_intraday.price + spot_vs_intraday.price15
    x = spot_vs_intraday  # [(spot_vs_intraday.index.year == 2022) & (spot_vs_intraday.index.month == 10)]
    x['s_15-b_spot'] = x.price15 - x.price.shift(-1)

    x_index_after_buy = (x.index.hour > 3) & (x.index.hour < 7) | ((x.index.hour > 14) & (x.index.hour < 18))
    x_index_after_sell = (x.index.hour > 8) & (x.index.hour < 13) | ((x.index.hour > 19) | (x.index.hour < 2))

    x_after_buy = (x.price15 - x.price.shift(-1))[x_index_after_buy]
    x_after_sell = (-x.price15 + x.price.shift(-1))[x_index_after_sell]
    x_after_buy_da = (- x.price15.shift(-1) + x.price)[x_index_after_buy]
    x_after_sell_da = (x.price15.shift(-1) - x.price)[x_index_after_sell]

    # for loop to search for the best moving average length for the id_da strategy. checking for 1 to 7 now

    lista = []
    shift = dt.timedelta(days=0)
    test_shift = shift + dt.timedelta(days=0)
    boolean_list_after_buy = []
    boolean_list_after_sell = []

    df = pd.DataFrame()
    for date in pd.date_range(date_from, date_to - test_shift):
        after_buy = x_after_buy[(x_after_buy.index.date >= date) & (x_after_buy.index.date <= date + shift)]
        after_sell = x_after_sell[(x_after_sell.index.date >= date) & (x_after_sell.index.date <= date + shift)]

        after_buy = after_buy.groupby(after_buy.index.hour).mean()
        after_sell = after_sell.groupby(after_sell.index.hour).mean()

        merged = pd.concat([after_buy, after_sell]).sort_index(ascending=True)
        new_index=[]
        for i in range(len(merged)):
            new_index.append(dt.datetime(year=date.year, month=date.month, day= date.day, hour=merged.index[i]))
        merged.index = new_index
        merged = merged.to_frame()
        df = df.append(merged)
        x=2

    df.rename(columns={0: 'price'}, inplace= True)
    fig, axs = plt.subplots(16)
    k = 0
    for group, i in df.groupby([df.index.hour]):
        axs[k].plot(i.index, i.iloc[:, 0])
        axs[k].set_title(f'{group}')
        k=k+1
    plt.show()
    return 0

def id_15_DA_production(date_from, date_to, frequency = '1H'):

    spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample(frequency,
                                                                                          label='left').mean().ffill()
    intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index(
        'datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
    intraday_auction_1h = intraday_auction.resample(frequency, label='left').agg({'price': 'mean'}).ffill()

    spot_vs_intraday = spot.join(intraday_auction_1h.rename(columns={'price': 'price15'}))
    spot_vs_intraday['15minus_spot'] = - spot_vs_intraday.price + spot_vs_intraday.price15
    x = spot_vs_intraday  # [(spot_vs_intraday.index.year == 2022) & (spot_vs_intraday.index.month == 10)]
    x['s_15-b_spot'] = x.price15 - x.price.shift(-1)

    x_index_after_buy = (x.index.hour > 3) & (x.index.hour < 7) | ((x.index.hour > 14) & (x.index.hour < 18))
    x_index_after_sell = (x.index.hour > 8) & (x.index.hour < 13) | ((x.index.hour > 19) | (x.index.hour < 2))

    x_after_buy = (x.price15 - x.price.shift(-1))[x_index_after_buy]
    x_after_sell = (-x.price15 + x.price.shift(-1))[x_index_after_sell]
    x_after_buy_da = (- x.price15.shift(-1) + x.price)[x_index_after_buy]
    x_after_sell_da = (x.price15.shift(-1) - x.price)[x_index_after_sell]

    # for loop to search for the best moving average length for the id_da strategy. checking for 1 to 7 now

    lista = []
    shift = dt.timedelta(days=3)
    test_shift = shift + dt.timedelta(days=0)
    boolean_list_after_buy = []
    boolean_list_after_sell = []
    df = pd.DataFrame()
    for date in pd.date_range(date_from, date_to - test_shift):
        after_buy = x_after_buy[(x_after_buy.index.date >= date - shift) & (x_after_buy.index.date <= date)]
        after_sell = x_after_sell[(x_after_sell.index.date >= date - shift) & (x_after_sell.index.date <= date)]

        after_buy = after_buy.groupby(after_buy.index.hour).mean()
        after_sell = after_sell.groupby(after_sell.index.hour).mean()

        merged = pd.concat([after_buy, after_sell]).sort_index(ascending=True)
        new_index = []
        for i in range(len(merged)):
            new_index.append(dt.datetime(year=date.year, month=date.month, day=date.day, hour=merged.index[i]))
        merged.index = new_index
        merged = merged.to_frame()
        df = df.append(merged)
        x = 2

        # test_after_buy = x_after_buy[(x_after_buy.index.date == date + test_shift)]  # test_after_sell = x_after_sell[(x_after_sell.index.date == date + test_shift)]  #  # c = 0  # for k in range(len(after_buy)):  #     if after_buy.values[k] * test_after_buy[k] > 0:  #  #         c = c + 1  # boolean_list_after_buy.append(c/len(after_buy))  # score_after_buy = c/len(after_buy)  #  # c = 0  # for k in range(len(after_sell)):  #     if after_sell.values[k] * test_after_sell[k] > 0:  #         c = c + 1  # boolean_list_after_sell.append(c / len(after_sell))  # score_after_sell = c / len(after_sell)  #  # x=2
    df.rename(columns={0: 'price'}, inplace=True)
    fig, axs = plt.subplots(16)
    k = 0
    for group, i in df.groupby([df.index.hour]):
        axs[k].plot(i.index, i.iloc[:, 0])
        axs[k].set_title(f'{group}')
        k = k + 1
    plt.show()
    return 0


def arima_trial(date_from, date_to):
    frequency = '1H'
    hours = day_ahead_strat.hours_to_trade()  # variables for after buy sell indeces

    spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample(frequency,
                                                                                          label='left').mean().ffill()
    intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index(
        'datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
    intraday_auction_1h = intraday_auction.resample(frequency, label='left').agg({'price': 'mean'}).ffill()

    spot_vs_intraday = spot.join(intraday_auction_1h.rename(columns={'price': 'price15'}))
    spot_vs_intraday['15minus_spot'] = - spot_vs_intraday.price + spot_vs_intraday.price15

    df = intraday_auction_1h
    freq = 24
    train, test = df.price[:-freq], df.price[-freq:]
    model = pmdarima.arima.ARIMA((3, 1, 0), (1, 0, 2, 24))
    model.fit(train)
    model.summary()
    pred = model.predict(freq)
    plt.plot(pred, c='b')
    plt.plot(test, c='r')
    plt.legend()
    plt.show()


    intraday_auction = intraday_auction
    intraday_auction_week = intraday_auction[intraday_auction.index.weekday <= 4]
    intraday_auction_weekend = intraday_auction[intraday_auction.index.weekday > 4]
    for hourminute, item in intraday_auction_week.groupby([intraday_auction_week.index.hour, intraday_auction_week.index.minute]):
        pass
    result = adfuller(intraday_auction_week.price.diff().dropna())

    for i, index in enumerate(pd.date_range(intraday_auction_week.index[0], intraday_auction_week.index[-1], freq='D')):
        arima_model = ARIMA(intraday_auction_week, order=(1, 1, 4))
        model = arima_model.fit()
        print(model.summary())

        pred_range = int(len(intraday_auction_week) * 0.05)
        y_train = intraday_auction_week.iloc[:int(len(intraday_auction_week) * 0.95)]
        y_pred = pd.Series(model.predict(96), index=intraday_auction_week.price[-96:].index)
        y_true = intraday_auction_week.price[-96:]

        mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true))
        rmse = np.mean((y_pred - y_true)**2)**0.5
        corr = np.corrcoef(y_pred, y_true)[0,1]
        pprint.pprint({'mape': mape, 'rmse': rmse, 'corr': corr})

        plt.plot(model.predict(y_pred))
        plt.plot(y_true)
        plt.show()

    return 0

date_from = dt.date(2021, 9, 1)
date_to = dt.date(2022, 12, 2)

#id_15 = intraday_auction_15(date_from, months_back=3, frequency='15min')
#features = id_15.intraday_auction_predict(date_from, date_to)
#arima_trial(date_from, date_to)
id_15_DA_production(date_from, date_to)
#intraday_dayahead_optimization(date_from, date_to)
#strat = id_15.intraday_auction_strategy(date_from, date_to)

spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to).resample('1H', label='left').mean().ffill()
intraday_auction = FCRData().get_spot_intraday_auction_15min_actual(date_from, date_to, area=str(23)).set_index('datetime').sort_index().drop(columns=['datetime_id', 'area_id'])
auctions = fcr_strat(date_from, date_to, frequency='4H').get_results()
spot_4h = spot.resample('4H',  label='left').agg({'price': 'mean'}).ffill()

