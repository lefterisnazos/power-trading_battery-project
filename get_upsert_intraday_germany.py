import concurrent
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import multiprocessing
import os
import pickle
from datetime import datetime, timedelta, date
from itertools import product
import wapi
from enscohelper.database import DBConnector

from dict_codes import *

""" get curves from wattsight, make adjustments, and then upsert. Intraday fundamentals 15min, intraday continuous 15min+hourly, intraday auction 15min """

config_file_path = 'ws_config.ini'
session = wapi.Session(config_file=config_file_path)

db = DBConnector('psql_ensco')
db.__enter__()


def wattsight_table_resample_fix(df):
    df = df.resample('60min', label='left').agg(
        {'datetime_id': 'first',
         'area_id': 'first',
         'production': 'mean'}
    )
    # to drop the minutes from datetime_id
    df['datetime_id'] = df.datetime_id.map(lambda x: x[:-2])

    return df


def data_filter(data, columns_renaming, fundamental_key=None):
    """
    :param data: dataframe
    :param columns_renaming: rename columns to match with sql database column names + drop any columns we dont need
    :return: clean dataframe ready for upsertion into the database
    """

    if len(data) > 0:
        drop_columns = [x for x in data.columns if x not in columns_renaming.keys()]
        data.drop(drop_columns, axis=1, inplace=True)
        data.rename(columns_renaming, axis=1, inplace=True)
        data = data.replace({"": None})
        data = data.replace({np.nan: None})
        data = data.replace({"-": None})

    # rename price column to the type of fundamental we gave
    if fundamental_key is not None:
        fundamental_names = ['consumption', 'residual_load', 'temperature', 'wind_production', 'hydro_production', 'pv_production']
        for i in fundamental_names:
            if i in fundamental_key:
                fundamental_key = i
                break
        try:
            data.rename(columns={'production': fundamental_key}, inplace=True)
        # for old wattsight tables, resample by 1 hour
        except Exception as e:
            data = wattsight_table_resample_fix(data)
            data.rename(columns={'production': fundamental_key.split('_actual')[0]}, inplace=True)

    return data


def get_key(dictionary, val):
    """
    :param dictionary:
    :param val: respective value of a certain key of the dict
    :return: given the value, returns the respective key. will produce error if there are dupblicate values.
    """
    for key, value in dictionary.items():
        if val == value:
            return f'{key}'


def get_intraday(date_from, date_to, curve_dict, upsert=False, fundamentals=False, columns_renaming_specific=None):
    """
    :param date_from: string val. example : '2022-08-02T19:00Z'. This is the string format from wattsight too
    :param date_to:
    :param curve: should be in a dictionary form, that each value contains the chart's name from wattsight_data
    :return: list that contains the dataframes with needed structure
    """

    df = []
    for key, values in curve_dict.items():
        curve = session.get_curve(name=values)
        ts = curve.get_data(data_from=date_from, data_to=date_to)

        x = ts.to_pandas().tz_convert('CET')
        x.index = x.index.tz_localize(None)
        x = x.to_frame()
        x.rename(columns={x.columns[0]: ts.name}, inplace=True)

        df.append(x)

        """ if upsert is False we just get a list of dataframes, if True, make adjustments to upsert """
        if upsert is False:
            continue
        else:
            x.rename(columns={curve_dict[key]: 'price'}, inplace=True)
            x['datetime_id'] = None
            for i in range(len(x)):
                # check if we have hourly or 15 minute data to have the correct datetime_id
                if x.index[1] - x.index[0] == timedelta(minutes=15):
                    x['datetime_id'][i] = x.index[i].strftime("%Y%m%d%H%M")
                else:
                    x['datetime_id'][i] = x.index[i].strftime("%Y%m%d%H")
            x['datetime'] = x.index
            x['area_id'] = 23

            if fundamentals is True:
                if columns_renaming_specific is None:
                    x = data_filter(x, columns_remaining_fundamentals,
                                    fundamental_key=get_key(curve_dict, curve_dict[key]))
                else:
                    x = data_filter(x, columns_renaming_specific, fundamental_key=get_key(curve_dict, curve_dict[key]))
            else:
                x = data_filter(x, columns_remaining_intraday)

            try:
                db.upsert(get_key(curve_dict, curve_dict[key]), 'wattsight_new', x.to_dict(orient='records'))
                print(f'{get_key(curve_dict, curve_dict[key])} upserted')
            except:
                x.index[i]
                print(f"Day: {get_key(curve_dict, curve_dict[key])}, {date} FAILED")

    return df


def get_intraday_forecasts(date_from, date_to, curve_dict, upsert=False, period_upsert=False, fundamentals=False,
                           columns_renaming_specific=None):
    """
    USED FOR dictionary that has list as values. For forecasts for different prediction models.

    :param date_from: string val. example : '2022-08-02T19:00Z'. This is the string format from wattsight too
    :param date_to:
    :param curve: should be in a dictionary form, that each value contains the chart's name from wattsight_data
    :return: list that contains the dataframes with needed structure
    """
    df = []
    if period_upsert is False:
        date_to = date_from
    for key, values in curve_dict.items():
        # iterate through every different source-prediction eg, for ec00, ec00ens etc
        for model in values:
            # update forecast each day
            for issue_date in pd.date_range(date_from, date_to, freq='D'):
                curve = session.get_curve(name=model)
                if curve.curve_type == 'TAGGED_INSTANCES':
                    ts = curve.get_instance(issue_date=issue_date.strftime("%Y-%m-%d"), tag='Avg')
                elif curve.curve_type == 'INSTANCES':
                    if period_upsert is False:
                        ts = curve.get_latest(issue_date_from=issue_date.strftime("%Y-%m-%d"))
                    else:
                        ts = curve.get_instance(issue_date=issue_date.strftime("%Y-%m-%d"))
                else: # for time_series_curve
                    ts = curve.get_data(data_from=date_from, data_to=date_to)

                x = ts.to_pandas().tz_convert('CET')
                x.index = x.index.tz_localize(None)
                x = x.to_frame()
                x.rename(columns={x.columns[0]: ts.name}, inplace=True)

                df.append(x)

                """ if upsert is False we just get a list of dataframes, if True, make adjustments to upsert """
                if upsert is False:
                    continue
                else:
                    x.rename(columns={model: 'price'}, inplace=True)
                    x['datetime_id'] = None
                    for i in range(len(x)):
                        # check if we have hourly or 15 minute data to have the correct datetime_id
                        if x.index[1] - x.index[0] == timedelta(minutes=15):
                            x['datetime_id'][i] = x.index[i].strftime("%Y%m%d%H%M")
                        else:
                            x['datetime_id'][i] = x.index[i].strftime("%Y%m%d%H")
                    x['datetime'] = x.index
                    x['area_id'] = '23'
                    if fundamentals is True:
                        try:
                            x['source'] = ts.sources[0]
                        except Exception as e:
                            if 'ec00' in ts.name:
                                x['source'] = 'EC00'
                            elif 'ec00ens' in ts.name:
                                x['source'] = 'EC00Ens'
                            else:
                                x['source'] = None
                        x = data_filter(x, columns_renaming_specific,
                                        fundamental_key=get_key(curve_dict, curve_dict[key]))
                    else:
                        x = data_filter(x, columns_renaming_specific)

                    try:
                        if 'fcr' in curve.name:
                            db.upsert(get_key(curve_dict, curve_dict[key]), 'fcr', x.to_dict(orient='records'))
                        else:
                            db.upsert(get_key(curve_dict, curve_dict[key]), 'wattsight_new', x.to_dict(orient='records'))
                        print(f'{get_key(curve_dict, curve_dict[key])} upserted')
                    except:
                        datestring= issue_date.strftime("%Y-%m-%d")
                        print(f"Day: {get_key(curve_dict, curve_dict[key])}, {datestring} FAILED")
    return df


def intraday_automatic_upsertion_task_scheduler():
    """
    date_from, date_to: we update in an interval of 2 full days, because at 3pm cet on a current day, trading for the next day starts.
    :return: None. we want to run the script every maybe 5 minutes with task scheduler.
    """
    date_from = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # appropriate string format for wattsight_api
    date_to = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")

    get_intraday(date_from, date_to, curves_dict_intraday, upsert=True)
    get_intraday(date_from, date_to, curves_dict_fundamentals, upsert=True, fundamentals=True)
    get_intraday_forecasts(date_from, date_to, curves_dict_fundamentals_forecast, upsert=True, period_upsert=False, fundamentals=True, columns_renaming_specific=columns_remaining_fundamentals_forecast)
    get_intraday_forecasts(date.today(), date.today(), curves_dict_fcr, upsert=True, period_upsert=True, fundamentals=False,columns_renaming_specific=columns_remaining_fcr)


def intraday_automatic_upsertion():
    """
    date_from, date_to: we update in an interval of 2 full days, because at 3pm cet on a current day, trading for the next day starts.
    :return: None. we want to run the script every maybe 5 minutes with task scheduler.
    """
    date_from = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # appropriate string format for wattsight_api
    date_to = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    get_intraday(date_from, date_to, curves_dict_intraday, upsert=True) #intraday prices
    get_intraday(date_from, date_to, curves_dict_fundamentals, upsert=True, fundamentals=True) #intraday actual forecast
    get_intraday_forecasts(date_from, date_to, curves_dict_fundamentals_forecast, upsert=True, period_upsert=False, fundamentals=True, columns_renaming_specific=columns_remaining_fundamentals_forecast)
    get_intraday_forecasts(date.today(), date.today(), curves_dict_fcr, upsert=True, period_upsert=False, fundamentals=False,columns_renaming_specific=columns_remaining_fcr)



area = 'Germany_Luxemburg'
#date_from = '2018-01-01'  # appropriate string format for wattsight_api
#date_to = '2019-12-31'
# date__from = date(2021, 1, 1)
# date__to = date(2022, 1, 1)
#
# intraday_automatic_upsertion_task_scheduler()
intraday_automatic_upsertion()

# executor = ThreadPoolExecutor()
# futures = [executor.submit(get_intraday, date.strftime("%Y-%m-%d"), (date + timedelta(days=1)).strftime("%Y-%m-%d"), curves_dict_intraday, upsert=True)
#            for date in pd.date_range(date(2022, 7, 1), date(2022, 12, 31))]