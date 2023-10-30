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
                   start_date='2018-11-01',
                   end_date='2018-11-30',
                   backward_days=1,
                   forward_days=1,
                   max_rows=50,
                   rand_seed=45,
                   fire_confidence=['h', 'high']
                   ):

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
    return row['latitude'], row['longitude'], row['fire_date'], row['start_date'], row['end_date']


def process_row(config, row, out_dir, buffer_size=0.1, img_size=None, layer='2_TRUECOLOR', max_cc=0.3,
                image_format=MimeType.PNG, data_collection=DataCollection.SENTINEL2_L2A):
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


if __name__ == "__main__":
    data_collection = DataCollection.SENTINEL2_L2A
    csv_in_dir = rf"C:\wildfire\data\csv\recent"
    base_out_dir = rf"C:\wildfire\data\images"
    # longitude = -121.437222
    # latitude = 39.810278
    longitude = None
    latitude = None

    config = SHConfig()

    if not config.sh_client_id or not config.sh_client_secret or not config.instance_id:
        config.instance_id = os.getenv('INSTANCE_ID')

    download_layer(config,
                   data_collection,
                   layer='2_TRUECOLOR',
                   csv_in_dir=csv_in_dir,
                   base_out_dir=base_out_dir,
                   image_format=MimeType.JPG,
                   buffer_size=0.05,
                   img_size=(350, 350),
                   longitude=longitude,
                   latitude=latitude,
                   start_date='2016-01-01',     # used for date filtering
                   end_date='2023-10-30',
                   backward_days=11,
                   forward_days=11,
                   max_rows=1000,
                   rand_seed=13,
                   fire_confidence=['h', 'high']
                   )