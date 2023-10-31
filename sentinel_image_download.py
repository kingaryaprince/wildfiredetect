import logging
import os
import random
import time
from datetime import timedelta

import pandas as pd
from PIL import Image
from sentinelhub import (SHConfig,
                         WmsRequest,
                         MimeType,
                         DataCollection,
                         CustomUrlParam,
                         DownloadFailedException)

from utilities.convert2bbox import convert2bbox
from utilities.prepare_csv_data import prepare_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def extract_row_data(row):
    """ Extract necessary data from the row """
    return row['latitude'], row['longitude'], row['fire_date'], row['start_date'], row['end_date']


def mime2ext(image_format):
    """ Convert MIME type to file extension """
    mime_to_extension = {
        MimeType.JPG: 'jpg',
        MimeType.PNG: 'png',
        MimeType.TIFF: 'tif',
    }
    return mime_to_extension.get(image_format)


def process_row(config, row, out_dir, buffer_size=0.1, img_size=None, layer='TRUECOLOR', max_cc=0.3,
                image_format=MimeType.JPG, data_collection=DataCollection.SENTINEL2_L2A):
    """
    Process a given row of data (that is extracted from VIIRS fire data), and extract available images 
    based on specified parameters such as longitude, latitude, timeinterval.

    Parameters:
    - config (SHConfig): Configuration object for the Sentinel Hub request.
    - row (pd.Series): A single row from the dataframe containing columns such as 'latitude', 'longitude', 'fire_date', etc.
    - out_dir (str): Directory where the downloaded images will be saved.
    - buffer_size (float, optional): Buffer size for bounding box calculation around latitude and longitude. Defaults to 0.1.
    - img_size (tuple, optional): Image dimensions as (width, height). If not provided, default values will be derived.
    - layer (str, optional): Layer specification for the Sentinel Hub request. Defaults to 'TRUECOLOR'.
    - max_cc (float, optional): Maximum cloud coverage in fraction (0 to 1). Defaults to 0.3.
    - image_format (MimeType, optional): Format for the downloaded image. Defaults to MimeType.JPG.
    - data_collection (DataCollection, optional): Data collection source. Defaults to DataCollection.SENTINEL2_L2A.

    Returns:
    None

    """
    latitude, longitude, fire_date, adj_start_date, adj_end_date = extract_row_data(row)

    bbox = convert2bbox(longitude=longitude, latitude=latitude, buffer_size=buffer_size)
    time_interval = (adj_start_date.strftime('%Y-%m-%dT%H:%M:%S'), adj_end_date.strftime('%Y-%m-%dT%H:%M:%S'))

    try:
        wms_request = WmsRequest(
            config=config,
            data_collection=data_collection,
            layer=layer,
            bbox=bbox,
            time=time_interval,
            width=img_size[0],
            height=img_size[1],
            maxcc=max_cc,
            image_format=image_format,
            time_difference=timedelta(hours=4),
            custom_url_params={CustomUrlParam.QUALITY: 95, CustomUrlParam.SHOWLOGO: False},
        )

        dates = wms_request.get_dates()
        print(f"Dates: {dates} Time Interval: {time_interval} Lon: {longitude} Lat: {latitude}")

        if dates:
            LOGGER.info(
                f"There are {len(dates)} Sentinel-2 images available with cloud coverage less than {max_cc * 100.0:.2f}%"
            )

            longitude_str = str(longitude).replace('.', '_')
            latitude_str = str(latitude).replace('.', '_')

            images = wms_request.get_data()

            for i, (dt_val, img_array) in enumerate(zip(dates, images), start=1):
                fire_time = fire_date.strftime('%Y-%m-%dT%H-%M-%S')
                img_time = dt_val.strftime("%Y-%m-%dT%H-%M-%S")
                img = Image.fromarray(img_array)
                ext = mime2ext(image_format)

                file_name = rf"lon{longitude_str}_lat{latitude_str}_F{fire_time}_W{img_time}_{i}_{layer.lower()}.{ext}"

                full_path = os.path.join(out_dir, file_name)
                img.save(full_path)
                LOGGER.info(f"Image is saved as {full_path}")

        else:
            LOGGER.debug(f"No images with maxx cc under {max_cc * 100:.2f}% between {time_interval}")

    except DownloadFailedException as e:
        LOGGER.error(f"Error from Search catalog: {str(e)}")


def download_layer(config, csv_in_dir, base_out_dir, data_collection, 
                   start_date, end_date, backward_days, forward_days, layer,
                   buffer_size=0.05, img_size=(350,350), max_rows=1000, rand_seed=45,
                   longitude=None, latitude=None, image_format=MimeType.JPG, fire_confidence=['h', 'high', 'n', 'nominal']):
    
    """
    Extract VIIRS fire data from previously saved csv files based on certain specific parameters, 
    and download corresponding to satellite imagery.

    Args:
    - config (SHConfig): Configuration object for the SentinelHub service.
    - data_collection (DataCollection): Specifies the data source collection (e.g., SENTINEL2_L2A).
    - layer (str): Name of the Layer configured in SentinelHub Configurator, for the specific instance (e.g. TRUECOLOR, FIREMASK) .
    - csv_in_dir (str): Path to the input directory containing VIIRS fire data CSV files.
    - base_out_dir (str): Base directory where the downloaded images will be saved.
    - buffer_size (float): Buffer size used when calculating bounding box from a point coordinate.
    - img_size (tuple): Tuple (width, height) representing the desired size of the downloaded image.
    - start_date (str): The earliest acquisition date to consider when filtering data from the CSV files (in 'YYYY-MM-DD' format).
    - end_date (str): The latest acquisition date to consider when filtering data from the CSV files (in 'YYYY-MM-DD' format).
    - backward_days (int): Number of days to look backwards from acquisition date for the time interval when downloading image.
    - forward_days (int): Number of days to look forward from acquisition date for the time interval when downloading image.
    - max_rows (int): Maximum number of rows (data points) to process.
    - rand_seed (int): Seed for random number generation, to randomly extact the rows from the input CSV files.
    - longitude (float, optional): Specific longitude coordinate to narrow down the data preparation. Default (None) means no limit 
    - latitude (float, optional): Specific latitude coordinate to narrow down the data preparation. Default (None) means no limit
    - image_format (MimeType, optional): Desired format of the downloaded image. Defaults to JPG.
    - fire_confidence (list, optional): List of fire confidence levels to filter the data. Defaults to ['h', 'high', 'n', 'nominal'].

    Returns:
    - None

    Output:
    - Downloads and saves images to a folder under `base_out_dir`.

    Notes:
    - If the resultant DataFrame after data preparation is empty, the function logs an error and does nothing further.
    - Between processing each row (or downloading each image), a random sleep interval is added for throttling.
    """
    project_name = data_collection.name[-6:]
    out_dir = os.path.join(base_out_dir, project_name)
    os.makedirs(out_dir, exist_ok=True)

    df = prepare_data(
        csv_in_dir=csv_in_dir,
        backward_days=backward_days,
        forward_days=forward_days,
        max_rows=max_rows,
        rand_seed=rand_seed,
        start_date=start_date,
        end_date=end_date,
        longitude=longitude,
        latitude=latitude,
        fire_confidence=fire_confidence
    )

    LOGGER.debug(f"Qualified df: {df}")
    if not df.empty:
        for _, row in df.iterrows():
            process_row(config=config, row=row, out_dir=out_dir, buffer_size=buffer_size, layer=layer,
                        img_size=img_size, image_format=image_format)
            time.sleep(5 * random.randint(1, 5))
    else:
        LOGGER.error(f"Nothing is done. Dataframe is empty")


if __name__ == "__main__":
    config = SHConfig()
    if not all([config.sh_client_id, config.sh_client_secret, config.instance_id]):
        config.instance_id = os.getenv('INSTANCE_ID')

    download_layer(config=config, data_collection=DataCollection.SENTINEL2_L2A, layer='TRUECOLOR',
                   csv_in_dir=r"C:\wildfire\data\csv\recent", base_out_dir=r"C:\wildfire\data\images",
                   buffer_size=0.05, img_size=(350, 350), start_date='2023-01-01', end_date='2023-10-30',
                   backward_days=0, forward_days=1, max_rows=1000, rand_seed=13,
                   fire_confidence=['h', 'high', 'n', 'nominal'])
