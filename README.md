# Wildfire Detection Using Deep Learning and Sentinel-2 Images

This project leverages the power of Deep Learning and Sentinel-2 satellite imagery to detect wildfires. It consists of two main components: a Sentinel Image Downloader and a Convolutional Neural Network (CNN) based model for wildfire detection.

## Components

This program has two components, one that downloads fire (and nofire) related images, and another one that applies machine learning to the images to detect fire or nofire

### 1. Sentinel Image Downloader (`sentinel_image_download.py`)
This script downloads wildfire and non-wildfire images from Sentinel-2, utilizing parameters based on VIIRS data. It requires SentinelHub credentials and specific configurations, including environment variables.

**Usage:**
```bash
python sentinel_image_download.py backward_days=0 forward_days=1
```
*   Images are saved in `C:\wildfire\data\images\<collection-abbreviation>`.
*   The script downloads images from a range defined by `backward_days` and `forward_days` relative to the fire date (`fire_date` or `acq_date`).
*   For optimal results, download images with TRUECOLOR and FIREMASK configurations from SentinelHub.

  2. Wildfire Detection (ml\fire_detect_cnn.py)
A CNN-based deep learning model trained on labeled wildfire images (fire and nofire).

Usage:

### 2. Wildfire Detection (ml\fire_detect_cnn.py)
A CNN-based deep learning model trained on labeled wildfire images (fire and nofire).

**Usage:**

To train and test the model:
```bash
python fire_detect_cnn.py --path C:/wildfire/data/images/all --epochs 10 --batch 32 --optimizer adam --train
```

To test the model with an existing trained model:
```bash
python fire_detect_cnn.py --path C:/wildfire/data/images/all --test --model_path path/to/model
```

## Requirements

- **Python Version**: Python 3.x
- **Libraries**:
  - `sentinelhub`
  - `pandas`
  - `numpy`
  - `PIL` (Python Imaging Library)
  - `logging`
  - `argparse`
  - `os`

- **For the Deep Learning Model**:
  - `keras`
  - `numpy`
  - `matplotlib`
  - `sklearn` (scikit-learn)

## Setup
### Image Download:

If you already have labeled data, an image download is not required. If you need to download the SentinelHub images, using API as in this program 
1) Sign up for the SentinelHub login access.
2) Create an OAuth client 
3) The Image Download program retrieves user credentials and other secrets from environment variables. So, configure these in your system environment
4) Pass the required parameters and run the program
5) Organize the data into "fire" and "nofire" folders, based on time of occurrence of fire and visual inspection

Please review https://www.sentinel-hub.com/explore/eobrowser/user-guide/ for additional details on setting up SentinelHub

### Wildfire Detection:

This is the actual detection program that is trained and tested on the categorized images
Based on loss, the best model is saved in the specified path, so the test can be rerun many times without going through expensive training of the model.
A performance plot (accuracy and loss) is saved alongside the model.

## Note
Paths mentioned are defaults and should be modified based on actual requirements.
Ensure all directories exist or are created before running the scripts.
Respect image copyrights and terms of use when using SentinelHub or other services.
The sentinel_image_download.py script relies on prepare_csv_data for preprocessing VIIRS fire data.

## License
This project is licensed under the MIT License.

## Citation
The study utilized VIIRS and MODIS fire data to download corresponding satellite images via the SentinelHub platform. CSV file for the last several years 
(https://firms.modaps.eosdis.nasa.gov/download/).

## Acknowledgments

This research was significantly enhanced by the support of the Network of Resources (NoR) at the European Space Agency (ESA), whose sponsorship enabled extended access to the SentinelHub platform, crucial for acquiring the high-resolution, multi-spectral imagery central to this study.

Additionally, both the Sentinel-Hub and Earthdata forums were indispensable in providing assistance and resolving inquiries related to fire and imagery data.

