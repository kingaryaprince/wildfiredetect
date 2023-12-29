Wildfire Detection Using Deep Learning and Sentinel-2 Images

This project consists of two primary components:

Sentinel Image Downloader (sentinel_image_download.py) - Downloads wildfire and non-wildfire images from Sentinel-2 using specified parameters based on VIIRS data.

Wildfire Detection (ml\fire_detect_cnn.py) - A CNN based deep learning model that is trained on the downloaded images or other wildfire images labelled as fire and nofire

Requirements
Python 3.x

Required libraries:
pandas
numpy
PIL
sentinelhub
logging
random
time
datetime
os

For the deep learning model:
- keras
- numpy
- matplotlib
- sklearn
- argparse

Steps
1. Sentinel Image Download (not required if images already exist under fire and nofire folder)
Before running the script, make sure you've set up your SentinelHub credentials and you have the required configurations, including any environment variables are in place.

Run the downloader with:

python sentinel_image_download.py backward_days=0 forward_days=1
By default, images will be saved in the C:\wildfire\data\images\<collection-abbreviation> directory. 

The program will download all images available between backward_days before the fire_date (or acq_date) and forward_days after the fire_date for the specific location. 
If a negative value is passed to the forward_days, it gets subtracted from fire_date 
(e.g. backward_days=30 and forward_days=-7, will extract all available images from 30 to 7 days before the fire

For easy identification of fire images, it is better to download the images with TRUECOLOR and FIREMASK that are configured under the SntinelHub instance. 
Please refer to https://docs.sentinel-hub.com/api/latest/api/ogc/wms/for downloading SentinelHub images using Web Mapping Service (WMS)

If labeled data is not available, review and move images with fire to the "fire" folder, and rest to the "nofire" folder.

Before invoking the next program, ensure files are under path\fire and path\nofire
e.g.
C:\wildfire\data\images\all\fire
C:\wildfire\data\images\all\nofire

2. Wildfire Detection (fire_detect_cnn)
After collecting and categorizing the images, the next step is to train and test our model.

python fire_detect_cnn.py --path C:/wildfire/data/images/all --epochs 10 --batch 32 --optimizer adam --train

If passing --train, the program will train and test the model. 
However, when passing the --test, it will just test the model using an existing model. If specifying the test, make sure to pass the previously trained model path

During the training, the best model based on loss will be saved under the specified path (e.g., path/best_model_{optimizer}_{epochs}_{timestamp}.keras)

A performance plot (accuracy and loss) for the model during training and validation is saved alongside the model with .png extension.
Please note: The given paths are default and can be modified as needed. Ensure all directories exist or are created before running the scripts. 

Always remember to respect image copyrights and terms of use when using SentinelHub services.

Image download (sentinel_image_download.py) relies on  prepare_csv_data, which preprocesses all the VIIRS fire data saved under a specific directory (pointed by the csv_in_dir parameter). (e.g.
The processing involves filtering, transforming, and cleaning the data to make it more meaningful and easier to analyze. Each file has embedded DCOSTRING which provide additional details on the usage.

License
This project is licensed under the MIT License.
