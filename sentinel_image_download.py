import logging
import argparse
import os
import random
import time
import pandas as pd

from datetime import timedelta
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


def download_layer(config,
                   data_collection,
                   layer,
                   csv_in_dir,
                   base_out_dir,
                   image_format,
                   buffer_size,
                   img_size,
                   longitude,
                   latitude,
                   start_date='2010-01-01',
                   end_date='2023-12-31',
                   backward_days=1,
                   forward_days=1,
                   max_rows=100,
                   rand_seed=45,
                   fire_confidence=['h', 'n']
                   ):
    """
    Downloads the specified layer of satellite images within the given time frame and geographical bounds.

    Parameters:
    config: SHConfig object for Sentinel Hub configuration.
    data_collection: DataCollection object specifying the satellite data collection to use. Default: DataCollection.SENTINEL2_L2A
    layer: String specifying the layer to download, which is configured under the sentinel instance configuration.
    csv_in_dir: Path to the input CSV file containing the data.
    base_out_dir: Base directory where the images will be saved.
    image_format: Format in which to save the images.
    buffer_size (float): Buffer size for bounding box calculations.
    img_size: Tuple specifying the image size in pixels for width and height. Default (350, 350)
    longitude (float): Longitude of the center point. Default: None
    latitude (float): Latitude of the center point. Default: None
    start_date (str): Start date for the time interval (default is '2010-01-01').
    end_date (str): End date for the time interval (default is '2023-12-31').
    backward_days (int): Number of days to go back from fire-date when extracting images. Default 7
    forward_days (int): umber of days to go forward from fire-date when extracting images. Default 7
    max_rows (int): Maximum number of rows to process from the CSV.
    rand_seed (int): Random seed for data shuffling.
    fire_confidence: List of fire confidence levels to include. default ['h', 'n'].
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
        LOGGER.error(f"Nothing is done. Data frame is empty")


def extract_row_data(row: pd.Series):
    """
    Extracts the row of the DataFrame and return key attributes.

    :param row: A row from a pandas DataFrame.
    :return: A tuple containing latitude, longitude, fire date, adjusted start date, and adjusted end date.
    """
    return row['latitude'], row['longitude'], row['fire_date'], row['start_date'], row['end_date']


def process_row(config, row, out_dir, buffer_size=0.1, img_size=None, layer='2_TRUECOLOR', max_cc=0.3,
                image_format=MimeType.PNG, data_collection=DataCollection.SENTINEL2_L2A):
    """
    Processes each rows from fire data and extracts satellite images for the corresponding location and time interval.

    Parameters:
    config: SHConfig object for Sentinel Hub configuration.
    row (pd.DataFrame): A row from a pandas DataFrame containing the data for a single location and date.
    out_dir (str): Directory where the images will be saved.
    buffer_size (float): Buffer size for bounding box calculations (default is 0.1).
    img_size (tuple (float, float)): Tuple specifying the image size in pixels for width and height (default is None).
    layer (str): String specifying the layer to download (This layer name is what is configured in the SentinelHub for the specific instance).
    max_cc (float): Maximum cloud coverage allowed for the images (default is 30%).
    image_format: Format in which to save the images (default is MimeType.PNG).
    data_collection: DataCollection object specifying the satellite data collection to use (default is DataCollection.SENTINEL2_L2A).

    This function extracts the necessary data from the provided row, calculates the bounding box and time interval for the request,
    and issues a WmsRequest to the Sentinel Hub API to fetch the images. If images are available, they are saved to the specified directory.

    Returns:
    None. 
    matching SentinelHub images will saved in the specified location for further processing 
    """
    latitude, longitude, fire_date, adj_start_date, adj_end_date = extract_row_data(row)

    LOGGER.info(
        f"Fetching images for longitude={longitude}, latitude={latitude}, "
        f"fire_date={fire_date} start_date={adj_start_date}, end_date={adj_end_date}"
    )

    bbox = convert2bbox(longitude=longitude, latitude=latitude, buffer_size=buffer_size)
    time_interval = adj_start_date.strftime('%Y-%m-%dT%H:%M:%S'), adj_end_date.strftime('%Y-%m-%dT%H:%M:%S')

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
            LOGGER.debug(f"No images with maxx-cc under {max_cc * 100:.2f}% between {time_interval}")

    except DownloadFailedException as e:
        LOGGER.error(f"Error from Search catalog: {str(e)}")


def mime2ext(image_format):
    mime_to_extension = {
        MimeType.JPG: 'jpg',
        MimeType.PNG: 'png',
        MimeType.TIFF: 'tif',
    }

    return mime_to_extension.get(image_format)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download satellite images based on wildfire data.')

    # Add arguments here
    parser.add_argument('--data_collection', type=str, default='SENTINEL2_L2A', help='Satellite data collection to use')
    parser.add_argument('--csv_in_dir', type=str, required=True, help='Path to the input CSV directory')
    parser.add_argument('--base_out_dir', type=str, required=True, help='Base output directory for images')
    parser.add_argument('--layer', type=str, default='TRUECOLOR', help='Layer from Sentnelhub instance configuration')
    parser.add_argument('--image_format', type=str, default='JPG', choices=['JPG', 'PNG', 'TIFF'],
                        help='Image format to save')
    parser.add_argument('--buffer_size', type=float, default=0.1, help='Buffer size for bounding box calculations')
    parser.add_argument('--img_width', type=int, default=350, help='Image width in pixels')
    parser.add_argument('--img_height', type=int, default=350, help='Image height in pixels')
    parser.add_argument('--longitude', type=float, help='Longitude of the center point')
    parser.add_argument('--latitude', type=float, help='Latitude of the center point')
    parser.add_argument('--start_date', type=str, default='2000-01-01', help='Start date for the time interval')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the time interval')
    parser.add_argument('--backward_days', type=int, default=1, help='Days to go back from fire date or acq_date')
    parser.add_argument('--forward_days', type=int, default=1, help='Days to go forward from fire date or acq_date')
    parser.add_argument('--max_rows', type=int, default=100, help='Maximum number of rows to process')
    parser.add_argument('--rand_seed', type=int, default=45, help='Random seed for data shuffling')
    parser.add_argument("--fire_confidence", nargs='+', default=['h', 'n'],
                        help='List of fire confidence levels to filter')
    parser.add_argument('--instance_id', type=str,
                        help='Instance ID for Sentinel Hub if not set in environment variables')

    args = parser.parse_args()

    config = SHConfig()
    if not (config.sh_client_id and config.sh_client_secret and config.instance_id):
        config.instance_id = os.getenv('INSTANCE_ID')
        config.sh_client_id = os.getenv('SH_CLIENT_ID')
        config.sh_client_secret = os.getenv('SH_CLIENT_SECRET')

    download_layer(config, DataCollection[args.data_collection], layer=args.layer, csv_in_dir=args.csv_in_dir,
                   base_out_dir=args.base_out_dir, image_format=MimeType[args.image_format],
                   buffer_size=args.buffer_size, img_size=(args.img_width, args.img_height), longitude=args.longitude,
                   latitude=args.latitude, start_date=args.start_date, end_date=args.end_date,
                   backward_days=args.backward_days, forward_days=args.forward_days, max_rows=args.max_rows,
                   rand_seed=args.rand_seed, fire_confidence=args.fire_confidence)
