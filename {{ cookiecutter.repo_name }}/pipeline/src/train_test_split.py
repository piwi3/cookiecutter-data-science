import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# 0. Parse all arguments and do validation checks if needed

parser = argparse.ArgumentParser("train_test_split")
parser.add_argument("--dataset_input_path", type=str, help="Input path of the input dataset")
parser.add_argument("--train_output_path", type=str, help="Output path of the train set")
parser.add_argument("--test_output_path", type=str, help="Output path of the test set")

args = parser.parse_args()


# 1. Read the electricity load data into pandas dataframe and sort by time
data = pd.read_csv(args.dataset_input_path, index_col=0, sep=';', decimal=',')
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)


# 2. Next, we aggregate to hourly data. Due to the model’s size and complexity, we train 
# our model on 5 consumers only (for those with non-zero values).
data = data.resample('1h').mean().replace(0., np.nan)
earliest_time = data.index.min()
df=data[['MT_002',	'MT_004',	'MT_005',	'MT_006',	'MT_008' ]]


# 3. Now, we prepare our dataset for the TimeSeriesDataset format. Notice that each column 
# represents a different time-series. Hence, we ‘melt’ our dataframe, so that all 
# time-series are stacked vertically instead of horizontally. In the process, we 
# create our new features.

df_list = []

for label in df:

    ts = df[label]

    start_date = min(ts.fillna(method='ffill').dropna().index)
    end_date = max(ts.fillna(method='bfill').dropna().index)

    active_range = (ts.index >= start_date) & (ts.index <= end_date)
    ts = ts[active_range].fillna(0.)

    tmp = pd.DataFrame({'power_usage': ts})
    date = tmp.index

    tmp['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
    tmp['hours_from_start'] = tmp['hours_from_start'].astype('int')
  
    tmp['days_from_start'] = (date - earliest_time).days
    tmp['date'] = date
    tmp['consumer_id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

    #stack all time series vertically
    df_list.append(tmp)

time_df = pd.concat(df_list).reset_index(drop=True)


# 4. Now we split the data into train and test training set and save both datasets as parquet files
# in the regarding output paths

train_start = 1096  # to match the original paper
train_end = 1346    # to match the original paper

train_data = time_df[(time_df.days_from_start >= train_start) & (time_df.days_from_start < train_end)]
train_data.to_parquet(Path(args.train_output_path, "electrical_load_train.parquet"))

test_data = time_df[time_df.days_from_start >= train_end]
test_data.to_parquet(Path(args.test_output_path, "electrical_load_test.parquet"))