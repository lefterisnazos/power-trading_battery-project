import concurrent.futures
from concurrent.futures._base import wait
from multiprocessing import pool

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
from enscohelper.database import DBConnector
from xbid_data_getter import *


def final_processing(df: pd.DataFrame):
	df = df.replace({"": None})
	df.sort_index(ascending=True, inplace=True)
	df = df.astype(float)
	df.replace({np.nan: None}, inplace=True)
	df['timestamp'] = df.index

	return df


db = DBConnector('psql_ensco')
db.__enter__()

start_date = dt.date(2022,1,1)
end_date = dt.date(2022,12,31)
start_time = dt.datetime(start_date.year, start_date.month,start_date.day, 0, 0)
end_time = start_time + dt.timedelta(days=1)
start_time = pd.Timestamp(start_time).tz_localize('CET').tz_convert('UTC').tz_localize(None)
end_time = pd.Timestamp(end_time).tz_localize('CET').tz_convert('UTC').tz_localize(None)


data = xbid_data()
delivery_area = '10YDE-VE-------2'
order_spread = data.bid_ask_order_history(start_time - dt.timedelta(seconds=10), end_time)
markets_list1 = []

for key, market in (order_spread.items()):
	market_id = key.strftime("%Y-%m-%d|%H:%M").split('|')[1]
	markets_list1.append(market_id)

markets_list1 = list(set(markets_list1))
df_bid = pd.DataFrame(columns=markets_list1)
df_ask = pd.DataFrame(columns=markets_list1)

for key, market in (order_spread.items()):
	market_id = key.strftime("%Y-%m-%d|%H:%M").split('|')[1]
	for i, item in enumerate(market):
		str_timestamp = market[i]['timestamp']
		dt_timestamp = dt.datetime.strptime(str_timestamp, "%Y-%m-%dT%H:%M:%SZ")
		timestamp = pd.Timestamp(dt_timestamp).tz_localize('UTC').tz_convert('CET')
		if item['max_buy_price'] is not None and item['min_sell_price'] is not None:
			bid = item['max_buy_price'] / 100 * 0.99
			ask = item['min_sell_price'] / 100 * 1.01
		elif item['max_buy_price'] is None and item['min_sell_price'] is None:
			bid = None
			ask = None
		elif item['max_buy_price'] is None:
			bid = None
			ask = item['min_sell_price'] / 100 * 1.01
		elif item['min_sell_price'] is None:
			ask = None
			bid = item['max_buy_price'] / 100 * 0.99
		df_bid.loc[timestamp, market_id] = bid
		df_ask.loc[timestamp, market_id] = ask


df_ask = final_processing(df_ask)
df_bid = final_processing(df_bid)


try:
	db.upsert('xbid_asks', 'xbid_data', df_ask.to_dict(orient='records'))
	db.upsert('xbid_bids', 'xbid_data', df_bid.to_dict(orient='records'))
except Exception as e:
	print(e)

x=2




