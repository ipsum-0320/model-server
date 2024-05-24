import math
import datetime

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import copy
from scipy import stats
from chinese_calendar import is_holiday
from io import StringIO


def excel_processor(file):
    np.seterr(invalid='ignore')
    slope_flag = 0  # [0, 1, 2]，0 表示前后 10 个点，1 表示前 10 个点，2 表示后 10 个点。
    pd.set_option('mode.chained_assignment', None)

    date_list = []
    value_list = []

    df = pd.read_csv(file)

    for date in df['date']:
        date_list.append(date)

    for value in df['value']:
        value_list.append(value)

    for i in range(len(value_list)):
        if i == 0:
            if abs(value_list[i] - value_list[i + 1]) >= 100:
                # 异常数据，相邻时刻的实例数相差 100 及以上。
                value_list[i] = value_list[i + 1]
        elif i == len(value_list) - 1:
            if abs(value_list[i] - value_list[i - 1]) >= 100:
                value_list[i] = value_list[i - 1]
        else:
            boundary = (value_list[i - 1] + value_list[i + 1]) / 2
            if (value_list[i] > value_list[i - 1] and value_list[i] > value_list[i + 1] and abs(
                    value_list[i] - boundary) >= 100) or (
                    value_list[i] < value_list[i - 1] and value_list[i] < value_list[i + 1] and abs(
                value_list[i] - boundary) >= 100):
                value_list[i] = boundary

    df_map = {
        "date": date_list,
        "value": value_list
    }
    df = DataFrame(df_map)

    df['date'] = pd.to_datetime(df['date'])
    df_with_day_of_week = copy.deepcopy(df)
    df_with_day_of_week['day_of_week'] = df_with_day_of_week['date'].dt.day_name()

    df['isMonday'] = "0"
    df['isTuesday'] = "0"
    df['isWednesday'] = "0"
    df['isThursday'] = "0"
    df['isFriday'] = "0"
    df['isSaturday'] = "0"
    df['isSunday'] = "0"
    df['hour'] = "00"  # 用于记录小时特征，
    df['slope'] = "0"  # 用于记录斜率特征。
    df['range'] = "0"  # 用于记录极差。
    df['deviation'] = "0"  # 用于记录标准差。
    df['isRest'] = "0"

    df['value'] = df['value']
    df.drop("value", axis=1, inplace=True)

    for i, item in enumerate(df_with_day_of_week['day_of_week']):
        df["is{}".format(item)][i] = "1"

    for i, item in enumerate(df['date']):
        df['hour'][i] = item.hour
        datetime_item = datetime.date(item.year, item.month, item.day)
        if is_holiday(datetime_item):
            df['isRest'][i] = "1"

    boundary = len(df['value'])
    for i, item in enumerate(df['value']):
        if slope_flag == 0:
            x = list(range((i - 10 if i - 10 >= 0 else 0), (i + 11 if i + 11 <= boundary else boundary)))
            y = df['value'][(i - 10 if i - 10 >= 0 else 0):(i + 11 if i + 11 <= boundary else boundary)].values
        elif slope_flag == 1:
            x = list(range((i - 10 if i - 10 >= 0 else 0), (i + 1 if i + 1 <= boundary else boundary)))
            y = df['value'][(i - 10 if i - 10 >= 0 else 0):(i + 1 if i + 1 <= boundary else boundary)].values
        else:
            x = list(range(i, (i + 11 if i + 11 <= boundary else boundary)))
            y = df['value'][i:(i + 11 if i + 11 <= boundary else boundary)].values
        slope = stats.linregress(x, y).slope
        _range = np.max(y) - np.min(y)
        deviation = np.std(y)
        if math.isnan(slope):
            slope = 0
        df['slope'][i] = "{}".format(round(slope, 2))
        df['range'][i] = "{}".format(round(_range, 2))
        df['deviation'][i] = "{}".format(round(deviation, 2))

    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

