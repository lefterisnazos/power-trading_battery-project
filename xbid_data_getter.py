import concurrent.futures
from concurrent.futures._base import wait
from multiprocessing import pool
import multiprocessing
import numpy as np
import pandas as pd
import requests
import datetime as dt
import pytz
import json
import time
import threading
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from concurrent.futures import ThreadPoolExecutor
import collections
import logging


def time_str_formatter(time_str):
    # time_str = time_str.split('.')[0]
    # time_str = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
    time_str = time_str.split('.')[0]

    return pd.Timestamp(time_str)


def get_key(dictionary, val):
    """
    :param dictionary:
    :param val: respective value of a certain key of the dict
    :return: given the value, returns the respective key. will produce error if there are dupblicate values.
    """
    for key, value in dictionary.items():
        if val == value:
            return f'{key}'


def hours_to_shift():
    localtime = pytz.timezone('Europe/Berlin')
    is_daylight = localtime.localize(pd.Timestamp(dt.date.today()))
    if bool(is_daylight.dst()):
        return 2
    else:
        return 1


class xbid_data:

    def __init__(self, time_from=dt.datetime.utcnow(), time_to=dt.datetime.utcnow() + dt.timedelta(hours=4)):
        self.delivery_area = '10YDE-VE-------2'  # '10YDE-RWENET---I' '10YDE-EON------1' '10YDE-ENBW-----N'
        self.product = 'XBID_Quarter_Hour_Power'  # 'XBID_Hour_Power'
        self.time_from = time_from
        self.time_to = time_to

        url = 'https://auth.volueinsight.com/oauth2/token'
        data = {'grant_type': 'client_credentials'}
        response = requests.post(url, data=data, auth=('5GQp-aLDDYnyo-Wm831ll8UdkyQr3E3q', '36.yImX1xNMCsXwYEi2Qc0b6TqjtG81uvzhbIm.vlcUn7ts8UGrkcKlBOQHKS-Fkc7q0k_T1iyzZJ8_0lZuDco5a8ERpDXbE_meN'))
        token = response.json().get('access_token')
        self.last_token_time = dt.datetime.utcnow()
        self.headers = {'Authorization': 'Bearer {}'.format(token)}

    def renew_authentication(self):

        time_now = dt.datetime.utcnow()
        if time_now - self.last_token_time > dt.timedelta(minutes=45):
            url = 'https://auth.volueinsight.com/oauth2/token'
            data = {'grant_type': 'client_credentials'}
            response = requests.post(url, data=data,
                                     auth=('5GQp-aLDDYnyo-Wm831ll8UdkyQr3E3q', '36.yImX1xNMCsXwYEi2Qc0b6TqjtG81uvzhbIm.vlcUn7ts8UGrkcKlBOQHKS-Fkc7q0k_T1iyzZJ8_0lZuDco5a8ERpDXbE_meN'))
            token = response.json().get('access_token')
            self.last_token_time = dt.datetime.utcnow()
            self.headers = {'Authorization': 'Bearer {}'.format(token)}

    @staticmethod
    def scrap():
        x = requests.get('https://live.volueinsight.com/api/v3/orders/active/?exchange=Epex&contract-id=13166895&delivery-area=10YDE-VE-------2')
        return x

    def time_bounds_to_cet(self):
        localtime = pytz.timezone('Europe/Berlin')
        is_daylight = localtime.localize(pd.Timestamp(dt.datetime.now()))
        if bool(is_daylight.dst()):
            self.time_from = self.time_from - dt.timedelta(hours=2)
            self.time_to = self.time_to - dt.timedelta(hours=2)
        else:
            self.time_from = self.time_from - dt.timedelta(hours=1)
            self.time_to = self.time_to - dt.timedelta(hours=1)

    def get_delivery_area(self):
        url = 'https://live.volueinsight.com/api/v3/delivery-areas/'
        params = {'exchange': 'Epex'}
        response = requests.get(url, params=params, headers=self.headers)
        delivery_areas = response.json()

        return delivery_areas

    def get_contracts(self, time_from, time_to, custom_period=False):
        # date_end is to check whether we need to get contracts of the next day too.

        date_end = (time_to.date() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        url_contracts = 'https://live.volueinsight.com/api/v3/contracts/'
        if not custom_period:
            params_contracts = {'exchange': 'Epex', 'product': self.product, 'date-end': date_end}
        else:
            date_start = time_from.date().strftime("%Y-%m-%d")
            date_end = time_to.date().strftime("%Y-%m-%d")
            params_contracts = {'exchange': 'Epex', 'product': self.product, 'date-begin': date_start, 'date-end': date_end }
            custom = 'https://live.volueinsight.com/api/v3/contracts?product=XBID_Quarter_Hour_Power&date-begin=2023-02-01&date-end=2023-02-10'
        response_contracts = requests.get(url_contracts, params=params_contracts, headers=self.headers)
        try:
            contracts = response_contracts.json()
            contracts_info = {}

            for i in range(len(contracts)):
                value = [contracts[i]['contract_id'], time_str_formatter(contracts[i]['delivery_from']), time_str_formatter(contracts[i]['delivery_until'])]
                contracts_info[i] = value

            contracts_df = pd.DataFrame.from_dict(contracts_info, orient='index', columns=['contract_id', 'timestamp_from', 'timestamp_to']).\
                set_index('timestamp_from').drop(columns=['timestamp_to']).sort_index(ascending=True)
        except Exception as e:
            logging.error('contracts getter error')
            logging.error(e)

        shift_it = False # converting timestamps into cet mode, depending on daylight saving mode. if we want
        if shift_it is True:
            hour_shift = hours_to_shift()
            contracts_df.index = contracts_df.index.shift(hour_shift, freq='H')

        contracts_df = contracts_df[(contracts_df.index >= time_from) & (contracts_df.index < time_to)]

        return contracts_df

    def get_best_bid_ask_profile(self, time_from, time_to, quantity=3):
        self.renew_authentication()
        contracts = self.get_contracts(time_from, time_to)
        url_active_orders = 'https://live.volueinsight.com/api/v3/orders/active/'
        ask = {}
        bid = {}
        start = time.time()
        # we have to ensure we are not using contracts that have expired, hence the following statement. There is a big delay happening somewhere, havent found it yet.
        local_time_now = dt.datetime.utcnow()
        # contracts = contracts[contracts.index > local_time_now + dt.timedelta(minutes=7)]
        key_list = []
        for i in range(len(contracts)):
            key_list.append(int(str(contracts.index[i].hour) + str(contracts.index[i].minute // 15 + 1)))

        start_request = time.time()

        def get_request(i: int):  # i is the index of the conctracts dataframe
            params_active_orders = {'exchange': 'Epex', 'contract-id': str(contracts['contract_id'][i]), 'delivery-area': self.delivery_area}
            response_active_orders = None
            try:
                response_active_orders = requests.get(url_active_orders, params=params_active_orders, headers=self.headers, verify=False, timeout=8)
            except Exception as e:
                logging.basicConfig(filename='xbid_data_errors.log', filemode='a')
                logging.exception(e)
                logging.error(f'{dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}\n')
                logging.error(f'contracts_list:{contracts.iloc[0,[0]]}\n')
            active_orders = response_active_orders.json()
            return active_orders

        executor = ThreadPoolExecutor()
        futures = [executor.submit(get_request, i) for i, contract in enumerate(contracts.contract_id.values)]
        done, not_done = wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        for l, future in enumerate(futures):
            active_orders = future.result(timeout=5)
            buy_orders, sell_orders = [], []
            for v in range(len(active_orders)):
                if active_orders[v]['kind'] == 'Sell':
                    sell_orders.append(active_orders[v])
                else:
                    buy_orders.append(active_orders[v])
            avg_buy_price, avg_sell_price = 0, 0

            cum_quantity, price = 0, None
            k = 0
            while cum_quantity < quantity and k < len(buy_orders):
                try:
                    price = buy_orders[k]['price'] / 100
                except IndexError:
                    xbid_data.save_error(buy_orders, contracts, time_from, time_to, k)
                order_quantity = buy_orders[k]['quantity'] / 1000
                cum_quantity = cum_quantity + order_quantity
                if cum_quantity > quantity:
                    weight = (order_quantity - (cum_quantity - quantity)) / quantity
                else:
                    weight = order_quantity / quantity
                avg_buy_price = avg_buy_price + weight * price
                k += 1

            cum_quantity, price = 0, None
            k = 0
            while cum_quantity < quantity and k < len(sell_orders):
                try:
                    price = sell_orders[k]['price'] / 100
                except IndexError:
                    xbid_data.save_error(sell_orders, contracts, time_from, time_to, k)
                order_quantity = sell_orders[k]['quantity'] / 1000
                cum_quantity = cum_quantity + order_quantity
                if cum_quantity > quantity:
                    weight = (order_quantity - (cum_quantity - quantity)) / quantity
                else:
                    weight = order_quantity / quantity
                avg_sell_price = avg_sell_price + weight * price
                k += 1

            ask[l] = np.round(avg_sell_price, 2)
            bid[l] = np.round(avg_buy_price, 2)

        end = time.time()
        duration = end - start
        executor.shutdown()
        # some keys will be done faster, so we have to sort ask bid by key

        ask = dict(zip(key_list, list(ask.values())))
        bid = dict(zip(key_list, list(bid.values())))
        end_request = time.time
        return bid, ask, contracts

    def get_interval_best_bid_ask(self, time_from, time_to):
        contracts = self.get_contracts(time_from, time_to)

        url_order_price_spread = 'https://live.volueinsight.com/api/v3/orders/spread/'

        ask = {}
        bid = {}

        for i in range(len(contracts)):
            params_order_price_spread = {'exchange': 'Epex', 'contract-id': str(contracts['contract_id'][i]),
                                         'delivery-area': self.delivery_area}
            response_order_price_spread = requests.get(url_order_price_spread, params=params_order_price_spread,
                                                       headers=self.headers)
            order_price_spread = response_order_price_spread.json()
            for k in range(len(order_price_spread)):
                if order_price_spread[k]['min_sell_price'] is not None:
                    key = int(str(contracts.index[i].hour) + str(contracts.index[i].minute//15 + 1))
                    ask[key] = order_price_spread[k]['min_sell_price']/100
                    break
            for k in range(len(order_price_spread)):
                if order_price_spread[k]['max_buy_price'] is not None:
                    key = int(str(contracts.index[i].hour) + str(contracts.index[i].minute//15 + 1))
                    bid[key] = order_price_spread[k]['max_buy_price']/100
                    break

        return bid, ask

    def bid_ask_order_history(self, time_from, time_to):
        contracts = self.get_contracts(time_from, time_to)
        contracts = contracts.tz_localize('UTC').tz_convert('CET').tz_localize(None)

        order_price_spread = None
        url_order_price_spread = 'https://live.volueinsight.com/api/v3/orders/spread/'

        orders_for_selected_contracts = {}

        for i in range(len(contracts)):
            params_order_price_spread = {'exchange': 'Epex', 'contract-id': str(contracts['contract_id'][i]), 'delivery-area': self.delivery_area}
            response_order_price_spread = requests.get(url_order_price_spread, params=params_order_price_spread, headers=self.headers)
            order_price_spread = response_order_price_spread.json()
            orders_for_selected_contracts[contracts.index[i]] = order_price_spread

        return orders_for_selected_contracts

    @staticmethod
    def save_error(x: list, contracts, time_from, time_to,k):
        print('len of active_orders_list:', len(x))
        time_now = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        with open(f'error_{time_from.strftime("%Y%m%d%H%M")} - {time_to.strftime("%Y%m%d%H%M")}.txt', 'w') as fp:
            json.dump([time_from.strftime("%Y%m%d-%H-%M"), time_to.strftime("%Y%m%d-%H-%M")], fp)
            fp.write('\n')
            json.dump(f'contract_ids: {contracts}', fp)
            fp.write('\n')
            json.dump(f'the index of the contract of the current contract_sets is:, {k}', fp)
            fp.write('\n')
            json.dump(f'time of error: {time_now}', fp)
            print('wrote active orders error to directory')

    def get_trades_for_given_period(self, time_from, time_to, lookback_from_delivery_start=3, query_interval=2):
        dict1 = {}
        url_trades = 'https://live.volueinsight.com/api/v3/trades/'
        contracts = self.get_contracts(time_from, time_to, custom_period=True)
        for i in range(len(contracts)):
            start = contracts.index[i] - dt.timedelta(hours=lookback_from_delivery_start)
            end = contracts.index[i] - dt.timedelta(hours=lookback_from_delivery_start) + dt.timedelta(minutes=query_interval)
            params = {'exchange': 'Epex', 'contract-id': str(contracts['contract_id'][i]), 'delivery-area': self.delivery_area, 'date-begin': start, 'date-end': end}
            response = requests.get(url_trades, params=params, headers=self.headers)
            trades = response.json()
            dict1[contracts.index[i]] = trades
        return dict1
















