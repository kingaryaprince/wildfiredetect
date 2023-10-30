import logging
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def prepare_data(csv_in_dir: str, start_date: str, end_date: str, backward_days: int = 5, forward_days: int = 5,
                 max_rows: int = 10, rand_seed: int = 52, longitude: float = None, latitude: float = None,
                 filter_buffer: float = 0.1, fire_confidence: list = ['h', 'high']) -> pd.DataFrame:

    LOGGER.info(f"Filtering CSV from {csv_in_dir} between dates {start_date} and {end_date}")

    columns_to_read = ['latitude','longitude', 'acq_date', 'acq_time', 'confidence']
    df = filter_csv_by_date(csv_in_dir, start_date, end_date, fire_confidence, columns_to_read)
    LOGGER.debug(f"Shape after initial date filtering is: {df.shape} and Data is:\n{df}")

    if latitude or longitude:
        df = filter_by_coordinates(df, longitude, latitude, filter_buffer)

    # df = df.sample(frac=1, random_state=rand_seed).head(max_rows)     # Commented. we should filter towards end

    df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
    df['fire_date'] = pd.to_datetime(
        df['acq_date'].astype(str) + ' ' +
        df['acq_time'].str[:2] + ':' +
        df['acq_time'].str[2:]
    )

    # change it to acq date to keep only date portion
    df['start_date'] = df['fire_date'] - timedelta(days=backward_days)
    df['end_date'] = df['fire_date'] + timedelta(days=forward_days)

    LOGGER.debug(f"Sorting by acq_date and de-duping to keep most recent record for the same location")
    df.sort_values(by=['acq_date', 'acq_time'], ascending=[False, False], inplace=True)
    df.drop_duplicates(subset=['longitude', 'latitude', 'acq_date'], keep='first', inplace=True)

    LOGGER.debug(f"Shape after de-dup for co-ordinates and acq_date {df.shape} and Data is:\n{df}")

    # Pick 10x random rows than required and pass it to drop_nearby, as passing the entire data set is not efficient
    df = df.sample(frac=1, random_state=rand_seed).head(max_rows * 10)    # newly added
    LOGGER.debug(f"Shape after initial sampling after de-dup {df.shape} and Data is:\n{df}")

    nearby_dist = 0.0001               # Set to a proximity to minimize the close by locations
    df = drop_nearby(df, nearby_dist) # Drop location that are close to each other from same fire time

    # df = df.sample(frac=1, random_state=rand_seed).reset_index(drop=True).head(max_rows)    # added new
    df = df.sample(frac=1, random_state=rand_seed).head(max_rows)    # removed reset index

    LOGGER.debug(f"Shape after final sample {max_rows} rows is {df.shape} and Data is:\n{df}")
    return df


def filter_csv_by_date(csv_in_dir: str, start_date: str, end_date: str, fire_confidence: list = ['h','high'], columns_to_read=None) -> pd.DataFrame:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    final_df = pd.DataFrame()

    csv_dir_path = Path(csv_in_dir)
    csv_files = [f for f in csv_dir_path.iterdir() if f.name.endswith('.csv')]

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, usecols=columns_to_read)
            df = df[df['confidence'].isin(fire_confidence)]
            df['acq_date'] = pd.to_datetime(df['acq_date'])
            df = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]

            final_df = pd.concat([final_df, df])  # Using concat for clarity and potential performance benefits

        except Exception as e:
            LOGGER.error(f"Error reading file {file_path}: {e}")

    final_df['longitude'] = final_df['longitude'].round(5)
    final_df['latitude'] = final_df['latitude'].round(5)
    final_df.reset_index(drop=True, inplace=True)

    LOGGER.debug("Converted CSV into dataframe format.")
    return final_df

def filter_by_coordinates(df, longitude, latitude, buffer):
    """Filter DataFrame based on provided coordinates and buffer."""

    north = round(min((latitude or 0) + buffer, 90.0), 5)
    south = round(max((latitude or 0) - buffer, -90.0), 5)
    east = round(min((longitude or 0) + buffer, 180.0), 5)
    west = round(max((longitude or 0) - buffer, -180.0), 5)

    df = df[
        (df['latitude'] >= south) & (df['latitude'] <= north) &
        (df['longitude'] >= west) & (df['longitude'] <= east)
        ]

    LOGGER.debug(f"Shape after adjusted co-ords: {west}, {south} ; {east} {north} including buffer: {buffer} is: {df.shape} and Data is:\n{df}")
    return df


def drop_nearby(df, nearby_dist=0.001, min_samples=1):
    # Convert DataFrame to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Using DBSCAN for clustering
    db = DBSCAN(eps=nearby_dist, min_samples=min_samples, metric='haversine').fit(np.radians(gdf[['latitude', 'longitude']].values))
    gdf['cluster'] = db.labels_

    LOGGER.debug(f"Shape before removing close by indices is {gdf.shape} and Data is:\n{gdf}")

    # Group by cluster and then pick the most recent point in each cluster
    # (assuming that the fire_date is in datetime format)
    gdf = gdf.sort_values(by='fire_date', ascending=False).groupby('cluster').head(1)

    # Drop the 'cluster' and 'geometry' columns before returning
    gdf = gdf.drop(columns=['cluster', 'geometry'])
    LOGGER.debug(f"Shape after drop-near-by with buffer {nearby_dist} is {gdf.shape} and Data is:\n{gdf}")

    return gdf

if __name__ == '__main__':
    csv_in_dir = rf"C:\wildfire\data\csv\recent"
    prepared_df = prepare_data(
        csv_in_dir=csv_in_dir,
        start_date='2016-08-01',
        end_date='2023-08-10',
        longitude=-119.417931,
        latitude=36.778259,
        max_rows=100,
        backward_days=21,
        forward_days=-7,
        fire_confidence=['h', 'high', 'n', 'nominal'],
        filter_buffer=0.1
    )

    # Set the display options
    pd.set_option('display.max_rows', None)  # or set a specific number like 1000
    pd.set_option('display.max_columns', None)  # or set a specific number like 50
    pd.set_option('display.width', None)  # set width to the maximum available screen width
    pd.set_option('display.max_colwidth', None)  # display full column data

    LOGGER.debug(prepared_df)
