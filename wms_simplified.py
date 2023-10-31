import os
from sentinelhub import (SHConfig,
                         MimeType,
                         DataCollection)
from sentinel_image_download import download_layer

if __name__ == "__main__":

    config = SHConfig()

    # Set the INSTANCE_ID, SH_CLIENT_ID and SH_CLIENT_SECRET in the environment  
    if not all([config.sh_client_id, config.sh_client_secret, config.instance_id]):
        config.instance_id = os.getenv('INSTANCE_ID')
        config.sh_client_id = os.getenv('SH_CLIENT_ID')
        config.sh_client_secret = os.getenv('SH_CLIENT_SECRET')
      

    download_layer(config,
                   data_collection=DataCollection.SENTINEL2_L2A,
                   layer='TRUECOLOR',
                   csv_in_dir=r"C:\wildfire\data\csv",
                   base_out_dir=r"C:\wildfire\data\images\2010",
                   image_format=MimeType.JPG,
                   buffer_size=0.05,
                   img_size=(350, 350),
                   longitude=None,
                   latitude=None,
                   start_date='2010-01-01',
                   end_date='2023-10-30',
                   backward_days=0,
                   forward_days=1,
                   max_rows=1000,
                   rand_seed=45,
                   fire_confidence=['h', 'high', 'n', 'nominal']
                   )
