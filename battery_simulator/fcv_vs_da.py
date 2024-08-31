import numpy as np
import pandas as pd
import datetime as dt
from enscohelper.datafeed import ActualData, ForecastData, FCRData
import math
import re

from battery_simulator.battery_simulator import Position, Battery


def da_corr_meanprice_spread(date_from=dt.date(2018, 11, 1), date_to=dt.date(2022, 12, 30), rolling_window=1):

    spread_list = []
    mean_price_list = []
    actual_spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to)
    for group, value in actual_spot.groupby(actual_spot.index.date):
        spread = value.max(axis=0)[0] - value.max(axis=1)[0]
        mean_price = value.mean(axis=0)[0]
        spread_list.append(spread)
        mean_price_list.append(mean_price)

    spread = pd.Series(spread_list).rolling(rolling_window).mean().dropna()
    mean = pd.Series(mean_price_list).rolling(rolling_window).mean().dropna()
    corr = np.corrcoef(mean.values, spread.values)

    relation_factor = mean/spread
    relation_factor.replace([np.inf, -np.inf], 0, inplace=True)
    relation_factor.dropna()
    relation_factor = np.mean(relation_factor.values)
    print ('corr:', corr, 'with a relation factor mean/spread of:', np.mean(mean/spread))
    return corr, np.mean(mean.values /spread.values)


class day_ahead_strat:

    def __init__(self, date_from, date_to, production=False):
        """
        :param date_from:
        :param date_to:
        :param type: produce the results of the strategy based on the forecasts if type is forecasts. Because we base our decision based on the forecasted margin, against fcr or in general
        """
        self.date_from = date_from
        self.date_to = date_to
        self.position = Position()
        self.battery = Battery(date_from)
        self.production = production
        if self.production is False:
            self.spot = ActualData().get_spot_prices('Germany_Luxemburg', date_from, date_to)
        else:
            self.spot = ForecastData().get_spot_prices('Germany', date_from, date_to)

    @staticmethod
    def hours_to_trade(date_from, date_to, hours_range=1, id_vs_da_production=False):

        forecast_spot = ForecastData().get_spot_prices('Germany', date_from, date_to)
        first_12hour = forecast_spot[forecast_spot.index.hour <= 11]
        second_12hour = forecast_spot[forecast_spot.index.hour > 11]
        sell_hours = [first_12hour.idxmax()[0].hour, second_12hour.idxmax()[0].hour]
        buy_hours = [first_12hour.idxmin()[0].hour, forecast_spot[(forecast_spot.index.hour >= sell_hours[0]) & (forecast_spot.index.hour <= sell_hours[1])].idxmin()[0].hour]

        if id_vs_da_production is True:
            return buy_hours, sell_hours

        buy_hours, sell_hours = [], []
        if hours_range == 1:
            buy_hours.append(3)
            sell_hours.append(8)
            #buy_hours.append(14)
            #sell_hours.append(19)

        return buy_hours, sell_hours

    def position_battery_update(self, spot: pd.DataFrame, i, quantity, type):
        if type == 'B':
            self.position.add_position(spot.price[i], quantity)
            self.position.save_trade(spot.index[i], quantity, spot.price[i], 'B')
        else:
            self.position.reduce_position(spot.price[i], quantity)
            self.position.save_trade(spot.index[i], quantity, spot.price[i], 'S')
            self.position.add_pnl_from_trade(spot.index[i], spot.price[i], quantity)

        self.battery.current_total_capacity(quantity, spot.index[i].date())

    def spot_profile_strategy(self, hours_range=1, max_quantity=3):
        spot = self.spot.copy()
        quantity = int(max_quantity/hours_range)
        buy_hours, sell_hours = day_ahead_strat.hours_to_trade(self.date_from, self.date_to)

        for i, datetime in enumerate(spot.index):
            if datetime.hour in buy_hours:
                self.position_battery_update(spot, i, quantity, 'B')

            if datetime.hour in sell_hours:
                self.position_battery_update(spot, i, quantity, 'S')

    def get_results(self, frequency='D'):
        self.spot_profile_strategy()
        return self.position.analytics(frequency)

    def get_scheduling(self):
        pass


class fcr_strat:

    def __init__(self, date_from, date_to, frequency='12H', production=False):
        self.date_from = date_from
        self.date_to = date_to
        self.production = production
        self.frequency = frequency
        self.fcr = self.get_data(date_from, date_to)

    def replace_hours(self, index: dt.datetime, product_name):
        return index.replace(hour=int(product_name[-5:-3]))

    def get_data(self, date_from, date_to):
        if self.production is False:
            fcr = FCRData().get_auction_results(date_from, date_to)
            fcr = fcr[(fcr['area'] == 'DE') & (fcr['tender_number'] == 1)]
            fcr['date'] = pd.to_datetime(fcr['date'])
            fcr.index = map(self.replace_hours, fcr.date, fcr.product_name)
            return fcr
        else:
            fcr = FCRData().get_fcr_price_forecast(date_from, date_to).set_index('datetime').\
                drop(columns=['area_id']).resample('4H', label='left', origin='end_day').agg({'price': 'sum'}).ffill()
            fcr = fcr[(fcr.index.date >= self.date_from) & (fcr.index.date <= self.date_to)]

            return fcr

    def get_results(self):
        if self.production is False:
            fcr = self.fcr.copy()
            fcr = fcr[['settlement_capacity_price']]
            fcr.rename(columns={'settlement_capacity_price': 'eur/mwh'}, inplace=True)
            fcr_frequency = fcr.groupby(pd.Grouper(freq=self.frequency)).sum()

            return [fcr, fcr_frequency]
        else:
            fcr = self.fcr.copy()
            fcr.rename(columns={'price': 'eur/mwh'}, inplace=True)
            fcr_frequency = fcr.groupby(pd.Grouper(freq=self.frequency)).sum()

            return [fcr, fcr_frequency]


class fcr_vs_da:

    def __init__(self, date_from, date_to, comparison_freq='12H', production=False):
        self.date_from = date_from
        self.date_to = date_to
        self.comparison_freq = comparison_freq
        self.fcr_win = [7, 3]
        self.strat_win = [3, 7]
        self.production = production

    def fcr_dayahead_production(self):
        date_from = self.date_from
        date_to = self.date_to

        day_ahead_forecast = day_ahead_strat(date_from, date_to, production=self.production).get_results(frequency=self.comparison_freq)[0]
        hours_to_trade = day_ahead_strat.hours_to_trade(date_from, date_to, id_vs_da_production=True)
        day_ahead_forecast = day_ahead_forecast[['eur/mwh']]
        fcr_per_12h = fcr_strat(date_from, date_to, production=self.production).get_results()[1]

        synthetic, battery_status = self.synthetic_results(day_ahead_forecast, fcr_per_12h, None)

        rev_col = 'total_combined_rev'
        print(synthetic)
        print(f'FCR_DA allocation: {synthetic[rev_col].sum()}', )
        print(f'with {battery_status[1]} used cycles')

        return synthetic.drop(columns=['allocation_success']), battery_status, hours_to_trade

    def fcr_dayahead_backtest(self):
        date_from = self.date_from
        date_to = self.date_to

        day_ahead = day_ahead_strat(date_from, date_to).get_results(frequency=self.comparison_freq)[0]
        day_ahead_forecast = day_ahead_strat(date_from, date_to). get_results(frequency=self.comparison_freq)[0]
        fcr_per_12h = fcr_strat(date_from, date_to).get_results()[1]

        synthetic, battery_status = self.synthetic_results(day_ahead, fcr_per_12h, day_ahead_forecast)

        rev_col = 'total_combined_rev'
        print(synthetic)
        print(f'synthetic strategy: {synthetic[rev_col].sum()}', )
        print(f'battery capacity at: {battery_status[0]}, with {battery_status[1]} used cycles')

        return synthetic, battery_status

    def synthetic_results(self, day_ahead: pd.DataFrame, fcr: pd.DataFrame, day_ahead_forecast):
        date_from = self.date_from
        fcr_win = self.fcr_win
        strat_win = self.strat_win

        synthetic = self._synthetic_df_creation(day_ahead, fcr, day_ahead_forecast, fcr_win, strat_win)
        battery_status = Battery(date_from).battery_life_simulator(synthetic.fcr_mw, synthetic.strat_mw)

        # price_dominance_error_fcr = len(
        #     synthetic[(synthetic.type == 'fcr') & (synthetic.allocation_success == 'fail')]) / len(
        #     synthetic[synthetic.type == 'fcr'])
        # price_dominance_error_dayahead = len(
        #     synthetic[(synthetic.type == 'day_ahead') & (synthetic.allocation_success == 'fail')]) / len(
        #     synthetic[synthetic.type == 'day_ahead'])

        return synthetic, battery_status

    @staticmethod
    def _synthetic_df_creation(day_ahead, fcr, day_ahead_forecast, fcr_win, strat_win):

        synthetic = pd.DataFrame(columns=['eur/mwh', 'type', 'fcr_mw', 'strat_mw', 'total_combined_rev', 'allocation_success'], index=day_ahead.index, data=None)
        if day_ahead_forecast is None:
            day_ahead_forecast = day_ahead
        for i in range(len(day_ahead)):
            if day_ahead_forecast['eur/mwh'][i] > fcr['eur/mwh'][i]:
                allocation_amount = strat_win
                synthetic['type'][i] = 'day_ahead'
                synthetic['total_combined_rev'][i] = day_ahead['eur/mwh'][i] * allocation_amount[0] + fcr['eur/mwh'][i] * allocation_amount[1]
                synthetic['fcr_mw'][i] = allocation_amount[0]
                synthetic['strat_mw'][i] = allocation_amount[1]
                synthetic['eur/mwh'][i] = (fcr['eur/mwh'][i] * allocation_amount[0] + day_ahead['eur/mwh'][i] * allocation_amount[1]) / 10
                synthetic['allocation_success'][i] = 'success' if day_ahead['eur/mwh'][i] > fcr['eur/mwh'][i] else 'fail'
            else:
                allocation_amount = fcr_win
                synthetic['type'][i] = 'fcr'
                synthetic['total_combined_rev'][i] = fcr['eur/mwh'][i] * allocation_amount[0] + day_ahead['eur/mwh'][i] * allocation_amount[1]
                synthetic['fcr_mw'][i] = allocation_amount[0]
                synthetic['strat_mw'][i] = allocation_amount[1]
                synthetic['eur/mwh'][i] = (fcr['eur/mwh'][i] * allocation_amount[0] + day_ahead['eur/mwh'][i] * allocation_amount[1]) / 10
                synthetic['allocation_success'][i] = 'success' if day_ahead['eur/mwh'][i] < fcr['eur/mwh'][i] else 'fail'

        return synthetic

    def compare_fcr_da_past_pnl(self):

        date_from = self.date_from
        date_to = self.date_to

        da = day_ahead_strat(date_from, date_to).get_results(frequency='12H')[0]
        fcr = fcr_strat(date_from, date_to, frequency='12H').get_results()[1]

        da = da.iloc[:, [2]].rename(columns={'eur/mwh': 'da_euw_mwh'})
        fcr= fcr.rename(columns={'eur/mwh': 'fcr_euw_mwh'})

        fcr = fcr.join(da)
        fcr['better_market'] = None
        for i in range(len(fcr)):
            if fcr.iloc[i, 0] >= fcr.iloc[i, 1]:
                fcr['better_market'][i] = 'fcr'
            else:
                fcr['better_market'][i] = 'da'


# a = day_ahead_strat(dt.date(2022,1,1), dt.date(2022,7,1))
# results = a.get_results()
# x=2