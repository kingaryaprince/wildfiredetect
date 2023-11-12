import logging
import argparse
import geopandas as gpd
import pandas as pd
import numpy as np

from datetime import timedelta
from pathlib import Path
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def prepare_data(csv_in_dir: str, start_date: str, end_date: str, backward_days: int = 5, forward_days: int = 5,
                 max_rows: int = 10, rand_seed: int = 52, longitude: float = None, latitude: float = None,
                 filter_buffer: float = 0.1, fire_confidence=None) -> pd.DataFrame:
    """
        Prepare and filter wildfire data based on various parameters.

        Parameters:
        csv_in_dir (str): Directory containing CSV files.
        start_date (str): Start date for filtering.
        end_date (str): End date for filtering.
        backward_days (int): Days to subtract for start date range.
        forward_days (int): Days to add for end date range.
        max_rows (int): Maximum rows to include in the final DataFrame.
        rand_seed (int): Seed for random sampling.
        longitude (float): Longitude for geographic filtering.
        latitude (float): Latitude for geographic filtering.
        filter_buffer (float): Buffer distance for geographic filtering.
        fire_confidence (list): List of fire confidence levels to filter.

        Returns:
        pd.DataFrame: Filtered and processed DataFrame.
    """
    if fire_confidence is None:
        fire_confidence = ['h', 'high']
    LOGGER.info(f"Filtering CSV from {csv_in_dir} between dates {start_date} and {end_date}")

    columns_to_read = ['latitude', 'longitude', 'acq_date', 'acq_time', 'confidence']
    df = filter_by_date(csv_in_dir, start_date, end_date, fire_confidence, columns_to_read)
    LOGGER.debug(f"Initial data shape and content:\nShape: {df.shape}\nData:\n{df}")

    if latitude or longitude:
        df = filter_by_coordinates(df, longitude, latitude, filter_buffer)

    df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
    df['fire_date'] = pd.to_datetime(
        df['acq_date'].astype(str) + ' ' +
        df['acq_time'].str[:2] + ':' +
        df['acq_time'].str[2:]
    )
    # df['fire_date'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')

    # Derive start and end date to download images for
    df['start_date'] = df['fire_date'] - timedelta(days=backward_days)
    df['end_date'] = df['fire_date'] + timedelta(days=forward_days)

    LOGGER.debug(f"Sorting by acq_date and de-duping to keep most recent record for the same location")
    df.sort_values(by=['acq_date', 'acq_time'], ascending=[False, False], inplace=True)
    df.drop_duplicates(subset=['longitude', 'latitude', 'acq_date'], keep='first', inplace=True)

    LOGGER.debug(f"Shape after de-dup for co-ordinates and acq_date:\nShape: {df.shape}\nData:\n{df}")

    # Pick 10x random rows than required and pass it to drop_nearby, as passing the entire data set is not efficient
    df = df.sample(frac=1, random_state=rand_seed).head(max_rows * 10)  # newly added
    LOGGER.debug(f"Shape after initial sampling after de-dup:\nShape: {df.shape}\nData:\n{df}")

    nearby_dist = 0.001  # Set to a proximity to minimize the close by locations
    df = drop_nearby(df, nearby_dist)  # Drop location that are close to each other from same fire time

    # extract only portion of the rows indicated by max_rows
    df = df.sample(frac=1, random_state=rand_seed).head(max_rows)

    LOGGER.debug(f"Shape after final sample {max_rows} rows:\nShape: {df.shape}\nData:\n{df}")
    return df


def filter_by_date(csv_in_dir, start_date, end_date, fire_confidence,
                   columns_to_read=None) -> pd.DataFrame:
    """
    Filter CSV files by date range and fire confidence.

    Parameters:
    csv_in_dir (str): Directory containing CSV files.
    start_date (str): Start date in YYYY-MM-DD format for filtering.
    end_date (str): End date for filtering.
    fire_confidence (list): List of fire confidence levels to filter.
    columns_to_read (list): Columns to read from CSV files.

    Returns:
    pd.DataFrame: Combined and filtered DataFrame from all CSV files.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    csv_dir_path = Path(csv_in_dir)
    csv_files = [f for f in csv_dir_path.iterdir() if f.name.endswith('.csv')]

    # Using a generator expression for efficient reading
    all_dfs = (pd.read_csv(file, usecols=columns_to_read) for file in csv_files)
    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df = final_df[final_df['confidence'].isin(fire_confidence)]
    final_df['acq_date'] = pd.to_datetime(final_df['acq_date'])
    final_df = final_df[(final_df['acq_date'] >= start_date) & (final_df['acq_date'] <= end_date)]

    final_df['longitude'] = final_df['longitude'].round(5)
    final_df['latitude'] = final_df['latitude'].round(5)
    final_df.reset_index(drop=True, inplace=True)

    LOGGER.debug("Converted CSV into dataframe format.")
    return final_df


def filter_by_coordinates(df, longitude, latitude, buffer):
    """
    Filter DataFrame based on provided coordinates and buffer.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    longitude (float): Longitude for center point.
    latitude (float): Latitude for center point.
    buffer (float): Buffer distance for geographic filtering.

    Returns:
    pd.DataFrame: DataFrame filtered based on geographic coordinates.
    """
    north = round(min((latitude or 0) + buffer, 90.0), 5)
    south = round(max((latitude or 0) - buffer, -90.0), 5)
    east = round(min((longitude or 0) + buffer, 180.0), 5)
    west = round(max((longitude or 0) - buffer, -180.0), 5)

    df = df[
        (df['latitude'] >= south) & (df['latitude'] <= north) &
        (df['longitude'] >= west) & (df['longitude'] <= east)
        ]

    LOGGER.debug(
        f"Shape after adjusted co-ords: {west}, {south} ; {east} {north} including buffer {buffer}:\nShape: {df.shape}\nData:\n{df}")
    return df


def drop_nearby(df, nearby_dist=0.001, min_samples=1):
    """
    Remove nearby points in DataFrame using DBSCAN clustering.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    nearby_dist (float): Distance threshold for nearby points.
    min_samples (int): Minimum samples for a cluster.

    Returns:
    pd.DataFrame: DataFrame with nearby points removed.
    """
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Using DBSCAN for clustering
    db = DBSCAN(eps=nearby_dist, min_samples=min_samples, metric='haversine').fit(
        np.radians(gdf[['latitude', 'longitude']].values))
    gdf['cluster'] = db.labels_

    LOGGER.debug(f"Shape and content before dropping near by locations:\nShape: {gdf.shape}\nData:\n{gdf}")

    # Group by cluster and then pick the latest fire_date in each cluster. For earliest, change False to True
    gdf = gdf.sort_values(by='fire_date', ascending=False).groupby('cluster').head(1)

    # Drop the 'cluster' and 'geometry' columns before returning
    gdf = gdf.drop(columns=['cluster', 'geometry'])
    LOGGER.debug(f"Shape and content after drop-near-by with buffer {nearby_dist}:\nShape: {gdf.shape}\nData:\n{gdf}")

    return gdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process wildfire data.")
    parser.add_argument("--csv_in_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--start_date", required=True, help="Start date for filtering from original input")
    parser.add_argument("--end_date", required=True, help="End date for filtering from original input")
    parser.add_argument("--longitude", type=float, default=None, help="Longitude for geographic filtering")
    parser.add_argument("--latitude", type=float, default=None, help="Latitude for geographic filtering")
    parser.add_argument("--max_rows", type=int, default=100, help="Maximum rows to include in the final DataFrame")
    parser.add_argument("--backward_days", type=int, default=7, help="Days to go back from fire date")
    parser.add_argument("--forward_days", type=int, default=7, help="Days to go forward from fire date")
    parser.add_argument("--fire_confidence", nargs='+', default=['h', 'n'],
                        help="List of fire confidence levels to filter")
    parser.add_argument("--filter_buffer", type=float, default=0.1, help="Buffer distance to get bounding box")

    args = parser.parse_args()

    prepared_df = prepare_data(
        csv_in_dir=args.csv_in_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        longitude=args.longitude,
        latitude=args.latitude,
        max_rows=args.max_rows,
        backward_days=args.backward_days,
        forward_days=args.forward_days,
        fire_confidence=args.fire_confidence,
        filter_buffer=args.filter_buffer
    )

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(prepared_df)
