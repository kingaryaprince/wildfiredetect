import os
from sentinelhub import (SHConfig,
                         MimeType,
                         DataCollection)
from wms_req import download_layer

if __name__ == "__main__":
    data_collection = DataCollection.SENTINEL2_L2A
    csv_in_dir = rf"C:\wildfire\data\csv\2010_2019"
    base_out_dir = rf"C:\wildfire\data\images\2010\possible-fire"

    config = SHConfig()

    if not config.sh_client_id or not config.sh_client_secret or not config.instance_id:
        config.instance_id = os.getenv('INSTANCE_ID')

    download_layer(config,
                   data_collection,
                   layer='2_TRUECOLOR',
                   csv_in_dir=csv_in_dir,
                   base_out_dir=base_out_dir,
                   image_format=MimeType.JPG,
                   buffer_size=0.2,
                   img_size=(350, 350),
                   longitude=None,
                   latitude=None,
                   start_date='2010-01-01',
                   end_date='2023-10-30',
                   backward_days=1,
                   forward_days=1,
                   max_rows=5000,
                   rand_seed=23,
                   fire_confidence=['h', 'high', 'n', 'nominal']
                   )