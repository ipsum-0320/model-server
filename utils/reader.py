import pandas as pd
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def convert(file, config, dry_run=False, start=0):
    df_raw = pd.read_csv(file)
    '''
  df_raw.columns: ['date', ...(other features), target feature]
  '''
    cols = list(df_raw.columns)
    cols.remove(config.target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [config.target]]
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    data = df_data.values
    if dry_run:
        df_stamp = df_raw[['date']][start:start + 180]
    else:
        # TODO: 需要更改为 df_raw[['date']][:]
        df_stamp = df_raw[['date']][1080:1260]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)

    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=config.freq)
    data_stamp = data_stamp.transpose(1, 0)
    if dry_run:
        data_x = data[start:start + 180]
        true_x = data[start + 180:start + 210]
        return data_x, data_stamp, true_x
    else:
        # TODO: 需要更改为 data[:]
        data_x = data[1080:1260]
        return data_x, data_stamp


def flatten_list(lst):
    flat_list = []
    for sublist in lst:
        if isinstance(sublist, list):
            flat_list.extend(flatten_list(sublist))
        else:
            flat_list.append(sublist)
    return flat_list
