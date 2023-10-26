import os

import pandas as pd


def filter_csv(csv_dir, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    final_df = pd.DataFrame()

    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_dir, filename)
            df = pd.read_csv(file_path)
            df = df[df['confidence'].isin(['h', 'high'])]
            df['acq_date'] = pd.to_datetime(df['acq_date'])
            df = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]
            # df = df.drop_duplicates(subset=['latitude', 'longitude', 'acq_date'])

            final_df = pd.concat([final_df, df], ignore_index=True)

    final_df.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)

    final_df.reset_index(drop=True, inplace=True)

    return final_df


if __name__ == '__main__':
    csv_dir = r"C:\Prince\Learn\ML\wildfire\data\csv\2023"
    df = filter_csv(csv_dir, start_date='2023-08-07', end_date='2023-08-15')
    print(df)
