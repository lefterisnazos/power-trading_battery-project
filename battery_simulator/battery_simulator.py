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
from enscohelper.datafeed import ActualData, ForecastData, FCRData


class Battery:

    def __init__(self, date_from):

        """ penalty for failing to deliver the capacity C. Percentage of the duration of the market interval, during which the battery is not available
                due to SoC exceeding  a minimum or maximum threshold

                Charging time assumption: 1 hour for 1 mw """

        self.max_cycles = 10000
        self.initial_max_capacity = 10 # mw max power that can be offered at the FCR market# total number of charge/discharge cycles. Battery life measured in cycles
        self.current_max_capacity = 10
        self.initial_max_fcr_capacity = self.initial_max_capacity - 2
        self.daily_loss_coefficient = 0.9999446516    # 365-root of 2% yearly power loss

        self.soc = 0 # mw total available power for any kind of trading/auction
        self.cycles_used = 0
        self.soc_message = ""

        self.rested = None  # hours since the battery lats rested, that was inactive and was charged/disarged to 50%
        self.penalty = None
        self.start_date = date_from

    def get_cycles(self, quantity):
        """
        :param quantity: amount of mw used to buy or sell
        :return: 1 cycle is completed after a 'full' discharge and then a 'full' charge, and vise versa
        """
        return 0.5 * quantity/self.current_max_capacity

    def update_soc(self, quantity, type):
        if type == 'B':
            if self.soc + quantity > self.current_max_capacity:
                self.soc_message = 'maximum capacity reached, try reduced'
                print(self.soc_message)
            else:
                self.soc = self.soc + quantity
                return True
        elif type == 'S':
            if self.soc - quantity < 0:
                self.soc_message = 'battery cant be discharged more, try reduced quantity'
                print(self.soc_message)
            else:
                self.soc = self.soc - quantity
                return True

    def get_soc(self):
        return self.soc

    @staticmethod
    def _cycles_capacity_coefficient(cycles):
        """
        :param cycles: total current cycles that have been used in battery
        :return: based on the cycles used, returns the capacity_coefficient that reduces the total capacity,
                For intermediate values we use a linear function between 2 key/values
        """
        capacity_based_on_cycles = {0: 1, 1000: 0.95, 2000: 0.9, 3000: 0.875, 4000: 0.84, 5000: 0.82, 6000: 0.78, 7000: 0.76,
                                    8000: 0.74}
        x1 = (cycles // 1000) * 1000
        x2 = x1 + 1000
        dy = (capacity_based_on_cycles[x2] - capacity_based_on_cycles[x1])
        dx = x2 - x1
        cycles_capacity_coefficient = dy * cycles / dx - x1 * dy / dx + capacity_based_on_cycles[x1]
        cycles_capacity_coefficient = round(cycles_capacity_coefficient, 4)

        return cycles_capacity_coefficient

    def current_total_capacity(self, quantity, current_date: dt.date):
        cycles = self.get_cycles(quantity)
        self.cycles_used = self.cycles_used + cycles

        cycles_coefficient = self._cycles_capacity_coefficient(self.cycles_used)
        time_coefficient = self.daily_loss_coefficient**(current_date-self.start_date).days
        total_coefficient = cycles_coefficient*time_coefficient

        self.current_max_capacity = self.initial_max_capacity * total_coefficient

    def battery_life_simulator(self, fcr: pd.Series, strat: pd.Series):
        """
        :param fcr: series with the interval allocations for fcr
        :param strat: series with the interval allocations for the strategy
        :return:
        """
        interval_coefficient = 12/24 # index range period
        fcr_daily_loss = 1
        fcr_loss = (((fcr.index[1] - fcr.index[0]).seconds/3600) / 24) * fcr_daily_loss

        for i in range(len(fcr)):
            # strat component, *2 because we each cell's quantity corresponds to a buy and a sell,
            cycles = self.get_cycles(strat[i]*2)
            self.cycles_used = self.cycles_used + cycles

            # fcr component
            fcr_loss_current = (0.35/6*(fcr[i]) - 0.35/6 + 0.35)*interval_coefficient
            self.cycles_used = self.cycles_used + fcr_loss_current

            cycles_coefficient = self._cycles_capacity_coefficient(self.cycles_used)
            self.current_max_capacity = self.initial_max_capacity * cycles_coefficient

        cycles_coefficient = self._cycles_capacity_coefficient(self.cycles_used)
        time_coefficient = self.daily_loss_coefficient ** (fcr.index[1] - fcr.index[0]).days
        total_coefficient = cycles_coefficient * time_coefficient
        self.current_max_capacity = self.initial_max_capacity * total_coefficient

        return [self.current_max_capacity, self.cycles_used]

    def battery_reward_fcr(self):
        pass


class Position:

    def __init__(self):
        self.buy_value = None  # total average bought value
        self.total_quantity = 0  # total live quantity
        self.total_profit = 0  # profit ammased by the trades
        self.df = pd.DataFrame(columns=['timestamp', 'quantity', 'price', 'type'])  # df to keep log of the trades
        self.trades_per_mw = pd.DataFrame(columns=['timestamp', 'pnl', 'quantity'])

    def add_position(self, price, quantity):
        self.total_quantity = self.total_quantity + quantity
        if self.buy_value is None:
            self.buy_value = price
        else:
            self.buy_value = self.buy_value * (self.total_quantity - quantity)/self.total_quantity + price/quantity

    def reduce_position(self, price, quantity):
        if self.total_quantity == 0:
            print('cant sell, cause we have 0 quantity')
        else:
            trade_pnl = quantity * (price - self.buy_value)
            self.total_profit = self.total_profit + trade_pnl
            self.total_quantity = self.total_quantity - quantity

    def save_trade(self, timestamp, quantity, price, type):
        # append each trade information to a dataframe containing all the buy & sells
        self.df.loc[(len(self.df))] = [timestamp, quantity, price, type]

    def add_pnl_from_trade(self, timestamp, price, quantity):
        # append the profit or loss from each buy-sell to a dataframe
        try:
            self.trades_per_mw.loc[(len(self.df))] = [timestamp, price - self.buy_value, quantity]
        except Exception as e:
            pass

    def analytics(self, frequency='D'):
        df = self.trades_per_mw.copy()
        df.index = df['timestamp']
        df.index = pd.to_datetime(df.index)
        df.drop(columns=['timestamp'], inplace=True)
        df['revenue'] = df.pnl * df.quantity

        month_stats = df.groupby(pd.Grouper(freq=frequency)).sum()
        month_stats.drop(columns=['pnl'], inplace=True)
        month_stats['eur/mwh'] = month_stats['revenue']/month_stats['quantity']
        total_revenue = month_stats.revenue.sum()

        return [month_stats, total_revenue]


class Position_xbid:

    def __init__(self):
        self.cum_profit = 0
        self.position = {}  # to keep the prices of the position
        self.position_type = {}  # to keep the type of the position. the use of position_type arose because of negative prices 1 for buy -1 for sell
        self.settled = {}
        self.settled_type = {}
        self.current_position_pnl = {}  # its the allocation part profit + the cost/profit amount from closing

    def get_position(self):
        return self.position

    def add_change_position(self, bid, ask, allocation):
        i = 0
        for key in bid:
            if allocation[i] == 1:
                self.position[key] = - ask[key]
                self.position_type[key] = 1
            elif allocation[i] == -1:
                self.position[key] = bid[key]
                self.position_type[key] = -1
            else:
                self.position[key] = 0
                self.position_type[key] = 0
            i = i + 1

    def check_and_settle_position(self, bid, allocation: list):
        flag = False
        # self.position empty check ensures that allocation list won't be empty.
        if self.position != {}:
            i = 0 # bid & ask keys have to be the same | allocation, bid, ask length has to be the same
            for key in self.position.copy():
                if key not in bid:
                    self.settled[key] = self.position[key]
                    # we use the settled_type variable to keep the settled actions of buy/sells, because if we have negative prices we cant get that info from the prices, from the self.position
                    self.settled_type[key] = self.position_type[key]
                    self.position.pop(key)
                    self.position_type.pop(key)
                    flag = True
                    i += 1
            return flag
        else:
            print('position empty')

    def settle_remaining_position(self, allocation: list):
        if self.position != {}:
            i = 0
            for key in self.position.copy():
                self.settled[key] = self.position[key]
                self.settled_type[key] = self.position_type[key]
                self.position.pop(key)
                self.position_type.pop(key)
                i += 1
        print ('finished: remaining position settled,we cant update allocation because of battery constraints')

    def get_current_position_pnl(self, bid, ask):
        for key, value in self.position.items():
            if value > 0:
                self.current_position_pnl[key] = self.position[key] - ask[key]
            elif value < 0:
                self.current_position_pnl[key] = bid[key] + self.position[key]
            else:
                self.current_position_pnl[key] = 0

        return self.current_position_pnl











